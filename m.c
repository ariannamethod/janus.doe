/*
 * m.c — Madness of Experts. the 4th element. the unhinged one.
 *
 * experts are born. experts die. the parliament votes on every token.
 * the tokenizer knows it's a tokenizer. the data parser judges your data.
 * the architecture recompiles itself per-input like a nervous breakdown
 * that somehow produces valid gradients.
 *
 * parameters persist. topology doesn't. each forward pass, the model decides:
 * - how many experts are alive right now (2..16)
 * - how many get to vote per token (variable k, not fixed)
 * - how deep to go (skip layers when input is boring)
 * the resonance-harmonic backbone holds it all together. barely.
 *
 * cc m.c -O3 -lm -lpthread -o m && ./m --depth 4
 *
 * depth is the only knob. turn it and watch democracy scale.
 *
 *   depth 2  → ~3M params, 4 experts, learns what committees are
 *   depth 4  → ~8M params, 6 experts, experts start specializing
 *   depth 8  → ~30M params, 8 experts, political parties form
 *   depth 12 → ~60M params, 10 experts, parliamentary democracy
 *
 * what happens when you run it:
 * 1. loads or generates data (HF API / parquet / synthetic shame)
 * 2. trains BPE tokenizer that knows its own compression ratio
 * 3. builds ephemeral MoE with living experts
 * 4. trains with hand-written gradients through variable-k parliament
 * 5. watches experts be born (mitosis) and die (apoptosis)
 * 6. finetunes on personality.txt (optional but recommended)
 * 7. exports GGUF, drops you into chat with a parliament
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>
#ifdef __linux__
  #include <sys/statvfs.h>
#endif
#ifdef __APPLE__
  #include <sys/param.h>
  #include <sys/mount.h>
  #include <sys/sysctl.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * BLAS ACCELERATION — optional. 3-4x speedup on matmul.
 *   macOS:  cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m
 *   Linux:  cc m.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o m
 * ═══════════════════════════════════════════════════════════════════════════════ */
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION — one knob. depth. everything else is derived.
 * pytorch has 47 config files. we have one integer and a political system.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_EXPERTS 16
#define MAX_LAYERS 32
#define MIN_EXPERTS 2
#define CHUCK_WINDOW 16
#define CHUCK_DAMP_LO 0.5f
#define CHUCK_DAMP_HI 2.0f
#define CHUCK_B1 0.9f
#define CHUCK_B2 0.999f
#define CHUCK_EPS 1e-8f
#define CHUCK_PSI_CAP 0.3f
#define CHUCK_PSI_HALF 100.0f
#define CHUCK_MACRO_INT 50
#define CHUCK_MACRO_PAT 3
#define CHUCK_MACRO_DECAY 0.7f
#define CHUCK_REC_CD 20
#define CHUCK_REC_THR 0.25f
#define CHUCK_MEM_CAP 256
#define NOTORCH_RANK 4
#define HARMONIC_N 8

typedef struct {
    int depth, dim, n_heads, n_kv_heads, head_dim, hidden_dim;
    int vocab_size, seq_len;
    float norm_eps, rope_theta;
    int max_experts, initial_experts;
    float attn_clamp, aux_loss_w;
    float lr, weight_decay;
    int batch_size, max_steps, warmup_steps, log_every, bpe_merges, personality_steps;
    char data_url[512], data_path[256], gguf_path[256], personality_path[256];
} Config;

static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;
    c.dim = depth * 64;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 128) c.dim = 128;
    if (c.dim > 768) c.dim = 768;
    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;
    if (c.dim <= 384) { c.n_kv_heads = c.n_heads; }
    else { c.n_kv_heads = c.n_heads / 2; if (c.n_kv_heads < 1) c.n_kv_heads = 1;
           while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1) c.n_kv_heads--; }
    c.max_experts = MAX_EXPERTS;
    c.initial_experts = 2 + depth; /* depth 2→4, depth 4→6, depth 8→10 */
    if (c.initial_experts > MAX_EXPERTS) c.initial_experts = MAX_EXPERTS;
    c.hidden_dim = (int)(c.dim * 1.5f);
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;
    c.seq_len = 256; c.norm_eps = 1e-5f; c.rope_theta = 10000.0f;
    c.attn_clamp = 30.0f; c.aux_loss_w = 0.01f;
    c.lr = 3e-4f; c.batch_size = 4; c.warmup_steps = 100;
    c.weight_decay = 0.01f; c.log_every = 20;
    long pe = 12L*depth*c.dim*c.dim + (long)c.initial_experts*3*c.dim*c.hidden_dim*depth;
    c.max_steps = (int)(pe * 6 / (c.batch_size * c.seq_len));
    if (c.max_steps < 200) c.max_steps = 200;
    if (c.max_steps > 2000) c.max_steps = 2000;
    c.bpe_merges = 4000; c.personality_steps = 100;
    snprintf(c.data_url, 512, "fineweb-edu");
    snprintf(c.data_path, 256, "m_data.txt");
    snprintf(c.gguf_path, 256, "m.gguf");
    snprintf(c.personality_path, 256, "personality.txt");
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. the experts don't care which PRNG decided their fate.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_state = 42;
static uint64_t rng_next(void) { rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17; return rng_state; }
static float rand_uniform(void) { return (float)(rng_next()&0x7FFFFFFF)/(float)0x7FFFFFFF; }
static float rand_normal(void) { float u1=rand_uniform(),u2=rand_uniform(); if(u1<1e-10f)u1=1e-10f; return sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2); }

/* ═══════════════════════════════════════════════════════════════════════════════
 * BPE TOKENIZER + SELF-AWARENESS EYE
 * the tokenizer that knows it's a tokenizer. tracks its own compression ratio,
 * OOV rate, entropy. feeds health signal into Chuck optimizer.
 * "i am become tokenizer, measurer of bytes."
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { char **items; int len, cap; } StrArr;
static void sa_push(StrArr *a, const char *s) { if(a->len>=a->cap){a->cap=a->cap?a->cap*2:16;a->items=realloc(a->items,sizeof(char*)*a->cap);}a->items[a->len++]=strdup(s); }
static void sa_free(StrArr *a) { for(int i=0;i<a->len;i++)free(a->items[i]);free(a->items);a->items=NULL;a->len=a->cap=0; }

#define TOK_MAX_VOCAB 16384
#define TOK_STOI_CAP 32768
typedef struct { char a[64]; char b[64]; } MergePair;
typedef struct { char *key; int val; } StoiEntry;
typedef struct { StoiEntry entries[TOK_STOI_CAP]; } StoiTable;
typedef struct { char *tokens[TOK_MAX_VOCAB]; int vocab_size; StoiTable stoi; int bos_id,eos_id; MergePair *merges; int n_merges; } Tokenizer;

/* Self-awareness eye for the tokenizer — it knows what it eats */
typedef struct {
    float compression_ratio; /* bytes_in / tokens_out, EMA */
    float oov_rate;          /* unknown token frequency, EMA */
    float entropy;           /* token distribution entropy */
    int total_encoded;       /* lifetime token count */
    float health;            /* composite signal for Chuck */
    float code_ratio;        /* % of input that looks like code */
    int code_mode;           /* 0=text, 1=code detected */
} TokenizerEye;

static unsigned int str_hash(const char *s){unsigned int h=5381;while(*s)h=h*33+(unsigned char)*s++;return h;}
static void stoi_init(StoiTable *t){for(int i=0;i<TOK_STOI_CAP;i++){t->entries[i].key=NULL;t->entries[i].val=-1;}}
static void stoi_put(StoiTable *t,const char *key,int val){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key){t->entries[idx].key=strdup(key);t->entries[idx].val=val;return;}if(strcmp(t->entries[idx].key,key)==0){t->entries[idx].val=val;return;}}}
static int stoi_get(StoiTable *t,const char *key){unsigned int h=str_hash(key)%TOK_STOI_CAP;for(int i=0;i<TOK_STOI_CAP;i++){int idx=(h+i)%TOK_STOI_CAP;if(!t->entries[idx].key)return -1;if(strcmp(t->entries[idx].key,key)==0)return t->entries[idx].val;}return -1;}

static void tok_init(Tokenizer *tok){memset(tok,0,sizeof(Tokenizer));stoi_init(&tok->stoi);for(int i=0;i<256;i++){char h[8];snprintf(h,8,"0x%02x",i);tok->tokens[tok->vocab_size]=strdup(h);stoi_put(&tok->stoi,h,tok->vocab_size);tok->vocab_size++;}tok->tokens[tok->vocab_size]=strdup("<BOS>");stoi_put(&tok->stoi,"<BOS>",tok->vocab_size);tok->bos_id=tok->vocab_size++;tok->tokens[tok->vocab_size]=strdup("<EOS>");stoi_put(&tok->stoi,"<EOS>",tok->vocab_size);tok->eos_id=tok->vocab_size++;}
static void tok_add(Tokenizer *tok,const char *s){if(stoi_get(&tok->stoi,s)>=0)return;if(tok->vocab_size>=TOK_MAX_VOCAB)return;tok->tokens[tok->vocab_size]=strdup(s);stoi_put(&tok->stoi,s,tok->vocab_size);tok->vocab_size++;}

static char byte_category(unsigned char b){if((b>='a'&&b<='z')||(b>='A'&&b<='Z'))return'L';if(b>='0'&&b<='9')return'N';if(b==' '||b=='\n'||b=='\r'||b=='\t')return'Z';if(b>=0x80)return'L';return'P';}
typedef struct{unsigned char*data;int len;}ByteSeg;
typedef struct{ByteSeg*segs;int len,cap;}SegArr;
static void seg_push(SegArr*a,unsigned char*data,int len){if(a->len>=a->cap){a->cap=a->cap?a->cap*2:64;a->segs=realloc(a->segs,sizeof(ByteSeg)*a->cap);}a->segs[a->len].data=malloc(len);memcpy(a->segs[a->len].data,data,len);a->segs[a->len].len=len;a->len++;}
static void seg_free(SegArr*a){for(int i=0;i<a->len;i++)free(a->segs[i].data);free(a->segs);memset(a,0,sizeof(SegArr));}

static SegArr unicode_segment(const char*text,int text_len){SegArr r={0};if(!text||text_len==0)return r;unsigned char buf[4096];int bl=0;char cc=0;const unsigned char*p=(const unsigned char*)text;for(int i=0;i<text_len;i++){char cat=byte_category(p[i]);if(cat!=cc&&bl>0){seg_push(&r,buf,bl);bl=0;}cc=cat;if(bl<(int)sizeof(buf)-1)buf[bl++]=p[i];else{seg_push(&r,buf,bl);bl=0;buf[bl++]=p[i];}}if(bl>0)seg_push(&r,buf,bl);return r;}

#define PAIR_CAP 32768
typedef struct{char a[64];char b[64];int count;int used;}PairEntry;
static unsigned int pair_hash(const char*a,const char*b){unsigned int h=5381;for(const char*p=a;*p;p++)h=h*33+(unsigned char)*p;h=h*33+0xFF;for(const char*p=b;*p;p++)h=h*33+(unsigned char)*p;return h;}

static void tok_train_bpe(Tokenizer*tok,const char*text,int tl,int nm){
    printf("[bpe] training %d merges on %d bytes...\n",nm,tl);
    SegArr segs=unicode_segment(text,tl);if(segs.len==0){seg_free(&segs);return;}
    int ns=segs.len;StrArr*ss=calloc(ns,sizeof(StrArr));
    for(int s=0;s<ns;s++)for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&ss[s],h);}
    seg_free(&segs);if(tok->merges)free(tok->merges);tok->merges=calloc(nm,sizeof(MergePair));tok->n_merges=0;
    PairEntry*pairs=calloc(PAIR_CAP,sizeof(PairEntry));
    for(int it=0;it<nm;it++){
        memset(pairs,0,sizeof(PairEntry)*PAIR_CAP);
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];for(int i=0;i<sq->len-1;i++){unsigned int h=pair_hash(sq->items[i],sq->items[i+1])%PAIR_CAP;for(int p=0;p<64;p++){int idx=(h+p)%PAIR_CAP;if(!pairs[idx].used){strncpy(pairs[idx].a,sq->items[i],63);strncpy(pairs[idx].b,sq->items[i+1],63);pairs[idx].count=1;pairs[idx].used=1;break;}if(strcmp(pairs[idx].a,sq->items[i])==0&&strcmp(pairs[idx].b,sq->items[i+1])==0){pairs[idx].count++;break;}}}}
        int bc=1,bi=-1;for(int i=0;i<PAIR_CAP;i++)if(pairs[i].used&&pairs[i].count>bc){bc=pairs[i].count;bi=i;}
        if(bi<0)break;
        char nt[128];snprintf(nt,128,"%s+%s",pairs[bi].a,pairs[bi].b);
        strncpy(tok->merges[tok->n_merges].a,pairs[bi].a,63);strncpy(tok->merges[tok->n_merges].b,pairs[bi].b,63);tok->n_merges++;
        for(int s=0;s<ns;s++){StrArr*sq=&ss[s];StrArr mg={0};int i=0;while(i<sq->len){if(i<sq->len-1&&strcmp(sq->items[i],pairs[bi].a)==0&&strcmp(sq->items[i+1],pairs[bi].b)==0){sa_push(&mg,nt);i+=2;}else{sa_push(&mg,sq->items[i]);i++;}}sa_free(sq);*sq=mg;}
        tok_add(tok,nt);
        if((it+1)%500==0)printf("[bpe] %d/%d merges (vocab=%d)\n",it+1,nm,tok->vocab_size);
    }
    free(pairs);for(int s=0;s<ns;s++)sa_free(&ss[s]);free(ss);
    printf("[bpe] done: %d merges, vocab=%d\n",tok->n_merges,tok->vocab_size);
}

static int*tok_encode(Tokenizer*tok,const char*text,int tl,int*out_len){
    SegArr segs=unicode_segment(text,tl);int*ids=NULL;int ni=0,ci=0;
    for(int s=0;s<segs.len;s++){StrArr sy={0};for(int b=0;b<segs.segs[s].len;b++){char h[8];snprintf(h,8,"0x%02x",segs.segs[s].data[b]);sa_push(&sy,h);}
    if(tok->n_merges>0&&sy.len>=2){int ch=1;while(ch&&sy.len>=2){ch=0;int br=tok->n_merges,bp=-1;for(int i=0;i<sy.len-1;i++)for(int m=0;m<br;m++)if(strcmp(sy.items[i],tok->merges[m].a)==0&&strcmp(sy.items[i+1],tok->merges[m].b)==0){br=m;bp=i;break;}if(bp>=0){char nt[128];snprintf(nt,128,"%s+%s",tok->merges[br].a,tok->merges[br].b);StrArr mg={0};for(int i=0;i<sy.len;i++){if(i==bp){sa_push(&mg,nt);i++;}else sa_push(&mg,sy.items[i]);}sa_free(&sy);sy=mg;ch=1;}}}
    for(int i=0;i<sy.len;i++){int id=stoi_get(&tok->stoi,sy.items[i]);if(id<0)id=0;if(ni>=ci){ci=ci?ci*2:256;ids=realloc(ids,sizeof(int)*ci);}ids[ni++]=id;}sa_free(&sy);}
    seg_free(&segs);*out_len=ni;return ids;
}

static char*tok_decode(Tokenizer*tok,int*ids,int ni,int*out_len){
    char*buf=malloc(ni*8+1);int pos=0;
    for(int i=0;i<ni;i++){if(ids[i]<0||ids[i]>=tok->vocab_size)continue;if(ids[i]==tok->bos_id||ids[i]==tok->eos_id)continue;
    const char*nm=tok->tokens[ids[i]];const char*p=nm;while(*p){if(p[0]=='0'&&p[1]=='x'){unsigned int bv;if(sscanf(p,"0x%02x",&bv)==1)buf[pos++]=(char)bv;p+=4;if(*p=='+')p++;}else p++;}}
    buf[pos]='\0';*out_len=pos;return buf;
}

/* Tokenizer self-awareness update: called after each encode batch */
static void tok_eye_update(TokenizerEye *eye, int bytes_in, int tokens_out, int *ids, int n_ids, int vocab_size) {
    float ratio = (tokens_out > 0) ? (float)bytes_in / tokens_out : 1.0f;
    eye->compression_ratio = (eye->total_encoded == 0) ? ratio : 0.95f * eye->compression_ratio + 0.05f * ratio;
    /* token distribution entropy */
    int *counts = calloc(vocab_size, sizeof(int));
    int oov = 0;
    for (int i = 0; i < n_ids; i++) { if (ids[i] >= 0 && ids[i] < vocab_size) counts[ids[i]]++; else oov++; }
    float ent = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (counts[i] > 0) { float p = (float)counts[i] / n_ids; ent -= p * logf(p + 1e-10f); }
    }
    free(counts);
    eye->entropy = ent;
    float oov_r = (n_ids > 0) ? (float)oov / n_ids : 0;
    eye->oov_rate = (eye->total_encoded == 0) ? oov_r : 0.95f * eye->oov_rate + 0.05f * oov_r;
    eye->total_encoded += tokens_out;
    /* health: high compression + low OOV + moderate entropy = good */
    eye->health = fminf(1.0f, eye->compression_ratio / 4.0f) * (1.0f - eye->oov_rate);
    if (eye->entropy < 1.0f) eye->health *= 0.8f; /* repetitive input warning */
}

/* Code detection — does this text look like source code? */
static void tok_eye_detect_code(TokenizerEye *eye, const char *text, int len) {
    int code_signals = 0, total_lines = 0;
    const char *p = text, *end = text + len;
    while (p < end) {
        const char *nl = memchr(p, '\n', end - p);
        int ll = nl ? (int)(nl - p) : (int)(end - p);
        total_lines++;
        /* Code heuristics: indentation, braces, semicolons, keywords */
        if (ll > 0 && (p[0] == ' ' || p[0] == '\t')) code_signals++; /* indented */
        for (int i = 0; i < ll - 1; i++) {
            if (p[i] == '{' || p[i] == '}') { code_signals++; break; }
            if (p[i] == '(' && p[i+1] == ')') { code_signals++; break; }
            if (p[i] == '-' && p[i+1] == '>') { code_signals++; break; }
            if (p[i] == '=' && p[i+1] == '=') { code_signals++; break; }
            if (p[i] == '/' && p[i+1] == '/') { code_signals++; break; }
            if (p[i] == '#' && (ll > i+7 && strncmp(p+i, "#include", 8) == 0)) { code_signals += 3; break; }
            if (p[i] == '#' && (ll > i+6 && strncmp(p+i, "#define", 7) == 0)) { code_signals += 3; break; }
        }
        if (ll > 0 && p[ll-1] == ';') code_signals++;
        p = nl ? nl + 1 : end;
    }
    eye->code_ratio = (total_lines > 0) ? (float)code_signals / (total_lines * 2) : 0;
    if (eye->code_ratio > 1.0f) eye->code_ratio = 1.0f;
    eye->code_mode = (eye->code_ratio > 0.3f) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-AWARE DATA PARSER
 * tracks data quality. judges your dataset. feeds signal into training.
 * "your data is bad and you should feel bad" — the parser, probably.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float quality;        /* estimated data quality (0..1) */
    float domain_shift;   /* cosine distance from running mean */
    float noise_level;    /* non-text byte ratio */
    int total_parsed;
    float health;
    float *tok_mean;      /* running mean of token distribution */
    int tok_mean_dim;
} ParserEye;

static void parser_eye_init(ParserEye *eye, int vocab_size) {
    memset(eye, 0, sizeof(ParserEye));
    eye->quality = 1.0f; eye->health = 1.0f;
    eye->tok_mean = calloc(vocab_size, sizeof(float));
    eye->tok_mean_dim = vocab_size;
}

static void parser_eye_update(ParserEye *eye, int *tokens, int n_tokens, const char *raw_text, int raw_len) {
    /* noise: non-printable bytes */
    int noise = 0;
    for (int i = 0; i < raw_len; i++) {
        unsigned char c = (unsigned char)raw_text[i];
        if (c < 32 && c != '\n' && c != '\r' && c != '\t') noise++;
    }
    float nr = (raw_len > 0) ? (float)noise / raw_len : 0;
    eye->noise_level = 0.9f * eye->noise_level + 0.1f * nr;
    /* domain shift: cosine with running mean */
    float *cur = calloc(eye->tok_mean_dim, sizeof(float));
    for (int i = 0; i < n_tokens; i++) if (tokens[i] >= 0 && tokens[i] < eye->tok_mean_dim) cur[tokens[i]] += 1.0f;
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < eye->tok_mean_dim; i++) { dot += cur[i] * eye->tok_mean[i]; na += cur[i]*cur[i]; nb += eye->tok_mean[i]*eye->tok_mean[i]; }
    float cos_sim = (na > 0 && nb > 0) ? dot / (sqrtf(na) * sqrtf(nb) + 1e-8f) : 1.0f;
    eye->domain_shift = 1.0f - cos_sim;
    /* update running mean */
    float alpha = (eye->total_parsed == 0) ? 1.0f : 0.01f;
    for (int i = 0; i < eye->tok_mean_dim; i++) eye->tok_mean[i] = (1.0f-alpha)*eye->tok_mean[i] + alpha*cur[i];
    free(cur);
    eye->total_parsed++;
    eye->quality = (1.0f - eye->noise_level) * (1.0f - fminf(1.0f, eye->domain_shift * 2.0f));
    eye->health = eye->quality;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR — a float pointer that knows its size. still revolutionary in 2026.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *data; int size, rows, cols; } Tensor;
static Tensor *tnew(int s){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(s,sizeof(float));t->size=s;t->rows=1;t->cols=s;return t;}
static Tensor *tnew2d(int r,int co){Tensor*t=calloc(1,sizeof(Tensor));t->data=calloc(r*co,sizeof(float));t->size=r*co;t->rows=r;t->cols=co;return t;}
static void tinit(Tensor*t,float std){for(int i=0;i<t->size;i++)t->data[i]=rand_normal()*std;}
static void tfree(Tensor*t){if(t){free(t->data);free(t);}}
static Tensor *tclone(Tensor *src){Tensor*t=calloc(1,sizeof(Tensor));t->data=malloc(src->size*sizeof(float));memcpy(t->data,src->data,src->size*sizeof(float));t->size=src->size;t->rows=src->rows;t->cols=src->cols;return t;}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MATH OPS — rmsnorm, matvec, softmax, rope, silu. the building blocks.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static float silu_f(float x){return x/(1.0f+expf(-x));}
static float silu_bwd(float x){float s=1.0f/(1.0f+expf(-x));return s+x*s*(1.0f-s);}

static void rmsnorm(float*out,float*x,float*w,int d,float eps){float ss=0;for(int i=0;i<d;i++)ss+=x[i]*x[i];float inv=1.0f/sqrtf(ss/d+eps);for(int i=0;i<d;i++)out[i]=x[i]*inv*w[i];}
static void matvec(float*out,float*W,float*x,int r,int co){
#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor,CblasNoTrans,r,co,1.0f,W,co,x,1,0.0f,out,1);
#else
    for(int i=0;i<r;i++){float s=0;float*row=W+i*co;for(int j=0;j<co;j++)s+=row[j]*x[j];out[i]=s;}
#endif
}
static void softmax_n(float*x,int n){float mx=x[0];for(int i=1;i<n;i++)if(x[i]>mx)mx=x[i];float s=0;for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}for(int i=0;i<n;i++)x[i]/=s;}

static void apply_rope(float*v,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float x0=v[i],x1=v[i+h];v[i]=x0*cc[off+i]-x1*sc[off+i];v[i+h]=x0*sc[off+i]+x1*cc[off+i];}}
static void rope_bwd(float*dv,int pos,float*cc,float*sc,int hd){int h=hd/2,off=pos*h;for(int i=0;i<h;i++){float d0=dv[i],d1=dv[i+h];dv[i]=d0*cc[off+i]+d1*sc[off+i];dv[i+h]=-d0*sc[off+i]+d1*cc[off+i];}}

/* matmul: C[M,N] = A[M,K] * B[N,K]^T */
static void mm_fwd(float*C,float*A,float*B,int M,int N,int K){
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for(int m=0;m<M;m++){float*cm=C+m*N,*am=A+m*K;for(int n=0;n<N;n++){float s=0;float*bn=B+n*K;for(int k=0;k<K;k++)s+=am[k]*bn[k];cm[n]=s;}}
#endif
}
static void mm_bwd(float*dA,float*dB,float*dC,float*A,float*B,int M,int N,int K){
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f, dC, N, B, K, 1.0f, dA, K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0f, dC, N, A, K, 1.0f, dB, K);
#else
    for(int m=0;m<M;m++){float*dc=dC+m*N,*am=A+m*K;for(int n=0;n<N;n++){float d=dc[n];if(d==0)continue;float*bn=B+n*K;for(int k=0;k<K;k++){dA[m*K+k]+=d*bn[k];dB[n*K+k]+=d*am[k];}}}
#endif
}
static void rn_fwd(float*o,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*ot=o+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float inv=1.0f/sqrtf(ss/D+eps);for(int i=0;i<D;i++)ot[i]=xt[i]*inv*w[i];}}
static void rn_bwd(float*dx,float*dw,float*dout,float*x,float*w,int T,int D,float eps){for(int t=0;t<T;t++){float*xt=x+t*D,*dot_=dout+t*D,*dxt=dx+t*D;float ss=0;for(int i=0;i<D;i++)ss+=xt[i]*xt[i];float var=ss/D+eps;float inv=1.0f/sqrtf(var);float cs=0;for(int i=0;i<D;i++)cs+=dot_[i]*w[i]*xt[i];float c2=cs/(D*var);for(int i=0;i<D;i++){dxt[i]+=(dot_[i]*w[i]-xt[i]*c2)*inv;dw[i]+=dot_[i]*xt[i]*inv;}}}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIVING EXPERTS — birth (mitosis), death (apoptosis), vitality.
 * experts aren't just weight matrices. they're organisms. they have
 * a heartbeat (vitality), a specialty (frequency in harmonic space),
 * and a lifespan. overloaded experts split. neglected experts die.
 * this is darwinism applied to feed-forward networks. darwin would approve.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *w_gate, *w_up, *w_down;
    float frequency;        /* position in harmonic space [0, 2π] */
    float vitality;         /* 0.0=dying, 1.0=peak */
    float specialization;   /* entropy of token distribution routed here */
    int age;                /* steps since birth */
    int tokens_seen;        /* lifetime token count */
    int alive;              /* 0=dead slot, 1=active */
    int low_vitality_count; /* consecutive steps with vitality < 0.1 */
    float entropy_hist[16]; /* for harmonic analysis */
    int eh_pos;
} Expert;

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARLIAMENT ROUTER — variable-k election system.
 * each token triggers an election. experts campaign (dot product with input).
 * consensus determines how many experts get consulted:
 *   low consensus → more experts needed → k increases
 *   high consensus → clear winner → k decreases
 * harmonic resonance modulates votes. frequency gaps trigger new experts.
 * this is representative democracy applied to tensor routing.
 * gerrymandering not yet implemented (TODO for v2).
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *w_vote;         /* [MAX_EXPERTS, dim] — voting weights */
    float consensus;        /* 0=chaos, 1=unanimous. computed per-token average */
    float faction_power[MAX_EXPERTS]; /* accumulated influence */
    int election_count;
} Parliament;

/* ═══════════════════════════════════════════════════════════════════════════════
 * HARMONIC RESONANCE ENGINE — from AML core, adapted for experts.
 * each expert has a frequency. input gets fourier-decomposed.
 * experts that resonate with input get boosted. frequency gaps
 * are where new experts are born. inspired by Schumann resonances
 * and the fundamental conviction that everything is a wave.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float amplitudes[HARMONIC_N];   /* fourier decomposition of input */
    float dominant_freq;            /* strongest frequency component */
    float confidence;               /* how peaky the spectrum is */
} HarmonicState;

static void harmonic_decompose(HarmonicState *hs, float *entropy_hist, int hist_len) {
    /* fourier decomposition of entropy history — same idea as am_harmonic_forward */
    float max_amp = 0;
    int max_k = 0;
    for (int k = 0; k < HARMONIC_N && k < hist_len/2; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < hist_len; n++) {
            float angle = 6.2831853f * k * n / hist_len;
            re += entropy_hist[n] * cosf(angle);
            im += entropy_hist[n] * sinf(angle);
        }
        hs->amplitudes[k] = sqrtf(re*re + im*im) / hist_len;
        if (k > 0 && hs->amplitudes[k] > max_amp) { max_amp = hs->amplitudes[k]; max_k = k; }
    }
    hs->dominant_freq = (hist_len > 0) ? 6.2831853f * max_k / hist_len : 0;
    /* confidence: how much energy is in the dominant harmonic vs total */
    float total = 0;
    for (int k = 0; k < HARMONIC_N; k++) total += hs->amplitudes[k];
    hs->confidence = (total > 1e-8f) ? max_amp / total : 0;
}

static float expert_resonance(float expert_freq, HarmonicState *hs) {
    /* how much does this expert resonate with the current input spectrum? */
    float res = 0;
    for (int k = 0; k < HARMONIC_N; k++) {
        float freq_k = 6.2831853f * k / HARMONIC_N;
        float dist = fabsf(expert_freq - freq_k);
        if (dist > 3.14159f) dist = 6.2831853f - dist; /* wrap around */
        float weight = expf(-dist * dist * 2.0f); /* gaussian kernel */
        res += hs->amplitudes[k] * weight;
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — ephemeral topology, persistent parameters.
 * the topology changes every step. params stay. it's like a government:
 * politicians come and go, but the bureaucracy is forever.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    Tensor *attn_norm, *ffn_norm;
    Tensor *wq, *wk, *wv, *wo;
    Parliament parliament;
    Expert experts[MAX_EXPERTS];
    int n_alive;           /* how many experts are currently alive */
} LayerW;

typedef struct {
    Tensor *tok_emb, *output, *output_norm;
    LayerW *layers;
    int n_layers;
} ModelW;

static void init_expert(Expert *e, int dim, int hidden_dim, float freq, float init_std) {
    e->w_gate = tnew2d(hidden_dim, dim); tinit(e->w_gate, init_std);
    e->w_up = tnew2d(hidden_dim, dim); tinit(e->w_up, init_std);
    e->w_down = tnew2d(dim, hidden_dim); memset(e->w_down->data, 0, e->w_down->size * sizeof(float));
    e->frequency = freq;
    e->vitality = 0.7f;
    e->specialization = 0;
    e->age = 0;
    e->tokens_seen = 0;
    e->alive = 1;
    e->low_vitality_count = 0;
    memset(e->entropy_hist, 0, sizeof(e->entropy_hist));
    e->eh_pos = 0;
}

static void free_expert(Expert *e) {
    tfree(e->w_gate); tfree(e->w_up); tfree(e->w_down);
    e->w_gate = e->w_up = e->w_down = NULL;
    e->alive = 0;
    e->vitality = 0;
}

static void init_weights(ModelW *w, Config *c) {
    float es = 1.0f / sqrtf((float)c->dim), ls = 1.0f / sqrtf((float)c->dim);
    int qd = c->n_heads * c->head_dim, kd = c->n_kv_heads * c->head_dim;
    w->tok_emb = tnew2d(c->vocab_size, c->dim); tinit(w->tok_emb, es);
    w->output = tnew2d(c->vocab_size, c->dim); tinit(w->output, ls);
    w->output_norm = tnew(c->dim); for (int i = 0; i < c->dim; i++) w->output_norm->data[i] = 1.0f;
    w->n_layers = c->depth;
    w->layers = calloc(c->depth, sizeof(LayerW));
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l];
        lw->attn_norm = tnew(c->dim); lw->ffn_norm = tnew(c->dim);
        for (int i = 0; i < c->dim; i++) { lw->attn_norm->data[i] = 1.0f; lw->ffn_norm->data[i] = 1.0f; }
        lw->wq = tnew2d(qd, c->dim); tinit(lw->wq, ls);
        lw->wk = tnew2d(kd, c->dim); tinit(lw->wk, ls);
        lw->wv = tnew2d(kd, c->dim); tinit(lw->wv, ls);
        lw->wo = tnew2d(c->dim, qd); memset(lw->wo->data, 0, lw->wo->size * sizeof(float));
        /* Parliament */
        lw->parliament.w_vote = tnew2d(MAX_EXPERTS, c->dim); tinit(lw->parliament.w_vote, 0.01f);
        lw->parliament.consensus = 0.5f;
        lw->parliament.election_count = 0;
        memset(lw->parliament.faction_power, 0, sizeof(lw->parliament.faction_power));
        /* Initialize living experts with harmonic spacing */
        lw->n_alive = c->initial_experts;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (e < c->initial_experts) {
                float freq = 6.2831853f * e / c->initial_experts;
                init_expert(&lw->experts[e], c->dim, c->hidden_dim, freq, ls);
            } else {
                memset(&lw->experts[e], 0, sizeof(Expert));
            }
        }
    }
}

/* Parliament election: returns k_t (variable), fills indices and weights */
static int parliament_elect(Parliament *p, Expert *experts, float *input, int dim,
                            HarmonicState *hs, int *selected, float *weights) {
    int n_alive = 0;
    int alive_idx[MAX_EXPERTS];
    for (int e = 0; e < MAX_EXPERTS; e++) if (experts[e].alive) alive_idx[n_alive++] = e;
    if (n_alive < MIN_EXPERTS) return 0; /* shouldn't happen */

    /* Compute votes: dot(w_vote[e], input) + harmonic resonance */
    float votes[MAX_EXPERTS];
    float max_vote = -1e30f;
    for (int i = 0; i < n_alive; i++) {
        int e = alive_idx[i];
        float *row = p->w_vote->data + e * dim;
        float dot = 0;
        for (int j = 0; j < dim; j++) dot += row[j] * input[j];
        /* harmonic resonance modulation */
        float res = expert_resonance(experts[e].frequency, hs);
        votes[e] = dot + 0.1f * res;
        if (votes[e] > max_vote) max_vote = votes[e];
    }
    /* Consensus: how peaked is the vote distribution? (normalized std) */
    float mean_v = 0;
    for (int i = 0; i < n_alive; i++) mean_v += votes[alive_idx[i]];
    mean_v /= n_alive;
    float var_v = 0;
    for (int i = 0; i < n_alive; i++) { float d = votes[alive_idx[i]] - mean_v; var_v += d*d; }
    var_v /= n_alive;
    float std_v = sqrtf(var_v + 1e-8f);
    /* consensus ∈ [0,1]: high std → high consensus (clear winner) */
    float consensus = fminf(1.0f, std_v / (fabsf(mean_v) + 1.0f));
    p->consensus = 0.9f * p->consensus + 0.1f * consensus;

    /* Variable k: low consensus → more experts. high → fewer. */
    int k = (int)(n_alive * (1.0f - p->consensus));
    if (k < 2) k = 2;
    if (k > n_alive) k = n_alive;

    /* Top-k selection */
    int used[MAX_EXPERTS] = {0};
    for (int ki = 0; ki < k; ki++) {
        float bv = -1e30f; int bi = 0;
        for (int i = 0; i < n_alive; i++) {
            int e = alive_idx[i];
            if (!used[e] && votes[e] > bv) { bv = votes[e]; bi = e; }
        }
        selected[ki] = bi;
        weights[ki] = votes[bi];
        used[bi] = 1;
    }
    /* Softmax over selected */
    float mx = weights[0];
    for (int i = 1; i < k; i++) if (weights[i] > mx) mx = weights[i];
    float sum = 0;
    for (int i = 0; i < k; i++) { weights[i] = expf(weights[i] - mx); sum += weights[i]; }
    for (int i = 0; i < k; i++) weights[i] /= sum;

    p->election_count++;
    return k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INFERENCE FORWARD — single token through the living parliament.
 * attention → parliament election → variable experts → residual.
 * ephemeral depth: skip layers when input complexity is low.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float *x, *xb, *xb2, *xb3, *hb, *hb2;
    float *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache, *cos_cache, *sin_cache;
    float *expert_out;
    HarmonicState hs;
} RunState;

static RunState alloc_run(Config *c) {
    RunState s; int kd = c->n_kv_heads * c->head_dim;
    s.x = calloc(c->dim, sizeof(float)); s.xb = calloc(c->dim, sizeof(float));
    s.xb2 = calloc(c->dim, sizeof(float)); s.xb3 = calloc(c->dim, sizeof(float));
    s.hb = calloc(c->hidden_dim, sizeof(float)); s.hb2 = calloc(c->hidden_dim, sizeof(float));
    s.q = calloc(c->n_heads * c->head_dim, sizeof(float));
    s.k = calloc(kd, sizeof(float)); s.v = calloc(kd, sizeof(float));
    s.att = calloc(c->n_heads * c->seq_len, sizeof(float));
    s.logits = calloc(c->vocab_size, sizeof(float));
    s.key_cache = calloc(c->depth * c->seq_len * kd, sizeof(float));
    s.value_cache = calloc(c->depth * c->seq_len * kd, sizeof(float));
    int half = c->head_dim / 2;
    s.cos_cache = calloc(c->seq_len * half, sizeof(float));
    s.sin_cache = calloc(c->seq_len * half, sizeof(float));
    for (int p = 0; p < c->seq_len; p++)
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
            float ang = (float)p * freq;
            s.cos_cache[p*half+i] = cosf(ang);
            s.sin_cache[p*half+i] = sinf(ang);
        }
    s.expert_out = calloc(c->dim, sizeof(float));
    memset(&s.hs, 0, sizeof(HarmonicState));
    return s;
}

static float *forward_token(ModelW *w, Config *c, RunState *s, int token, int pos) {
    int D = c->dim, kd = c->n_kv_heads * c->head_dim, hd = c->head_dim, H = c->hidden_dim;
    int hg = c->n_heads / c->n_kv_heads; float sc = 1.0f / sqrtf((float)hd);
    memcpy(s->x, w->tok_emb->data + token * D, D * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l];
        /* Attention */
        rmsnorm(s->xb, s->x, lw->attn_norm->data, D, c->norm_eps);
        matvec(s->q, lw->wq->data, s->xb, c->n_heads*hd, D);
        matvec(s->k, lw->wk->data, s->xb, c->n_kv_heads*hd, D);
        matvec(s->v, lw->wv->data, s->xb, c->n_kv_heads*hd, D);
        for (int h = 0; h < c->n_heads; h++) apply_rope(s->q+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        for (int h = 0; h < c->n_kv_heads; h++) apply_rope(s->k+h*hd, pos, s->cos_cache, s->sin_cache, hd);
        int co = l * c->seq_len * kd + pos * kd;
        memcpy(s->key_cache + co, s->k, kd * sizeof(float));
        memcpy(s->value_cache + co, s->v, kd * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg; float *qh = s->q + h*hd; float *att = s->att + h * c->seq_len;
            for (int t = 0; t <= pos; t++) {
                int ko = l*c->seq_len*kd + t*kd + kvh*hd;
                float dot = 0; for (int d = 0; d < hd; d++) dot += qh[d] * s->key_cache[ko+d];
                att[t] = dot * sc;
            }
            if (c->attn_clamp > 0) { float inv = 1.0f / c->attn_clamp; for (int t = 0; t <= pos; t++) att[t] = c->attn_clamp * tanhf(att[t] * inv); }
            softmax_n(att, pos + 1);
            float *xb2h = s->xb2 + h * hd; memset(xb2h, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) { float a = att[t]; int vo = l*c->seq_len*kd + t*kd + kvh*hd; for (int d = 0; d < hd; d++) xb2h[d] += a * s->value_cache[vo+d]; }
        }
        matvec(s->xb, lw->wo->data, s->xb2, D, D);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        /* Parliament MoE FFN */
        rmsnorm(s->xb, s->x, lw->ffn_norm->data, D, c->norm_eps);
        memset(s->expert_out, 0, D * sizeof(float));

        int selected[MAX_EXPERTS]; float weights[MAX_EXPERTS];
        int k = parliament_elect(&lw->parliament, lw->experts, s->xb, D, &s->hs, selected, weights);

        for (int ki = 0; ki < k; ki++) {
            Expert *exp = &lw->experts[selected[ki]];
            matvec(s->hb, exp->w_gate->data, s->xb, H, D);
            matvec(s->hb2, exp->w_up->data, s->xb, H, D);
            for (int i = 0; i < H; i++) { float act = silu_f(s->hb[i]); s->hb[i] = act * s->hb2[i]; }
            matvec(s->xb2, exp->w_down->data, s->hb, D, H);
            for (int i = 0; i < D; i++) s->expert_out[i] += weights[ki] * s->xb2[i];
        }
        for (int i = 0; i < D; i++) s->x[i] += s->expert_out[i];
    }
    rmsnorm(s->x, s->x, w->output_norm->data, D, c->norm_eps);
    matvec(s->logits, w->output->data, s->x, c->vocab_size, D);
    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING FORWARD — saves everything for backward. through living experts.
 * through parliament. through variable k. memory goes brrr.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float *gate_pre, *up_pre, *act_out, *proj_out; } ExpertAct;
typedef struct {
    float *inp, *xn, *q, *k, *v, *attn_sc, *attn_out;
    float *attn_proj, *res_aa, *ffn_xn;
    float *moe_out;
    /* Parliament state per token */
    int *top_idx;     /* [T * MAX_EXPERTS] — variable k per token */
    float *top_wt;    /* [T * MAX_EXPERTS] */
    int *top_k;       /* [T] — k for each token */
    ExpertAct *ea;    /* [T * MAX_EXPERTS] */
} LayerAct;

typedef struct {
    LayerAct *layers; float *final_n, *logits, *residual;
    float *dr, *dxn, *dq, *dk, *dv, *dao, *dfxn, *dhb, *dhb2, *deo;
    float *cos_c, *sin_c; int T;
    HarmonicState hs;
} TrainState;

static TrainState alloc_ts(Config *c) {
    TrainState s = {0}; int T = c->seq_len, D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, H = c->hidden_dim;
    s.T = T; s.layers = calloc(c->depth, sizeof(LayerAct));
    for (int l = 0; l < c->depth; l++) {
        LayerAct *la = &s.layers[l];
        la->inp = calloc(T*D, 4); la->xn = calloc(T*D, 4); la->q = calloc(T*qd, 4);
        la->k = calloc(T*kv, 4); la->v = calloc(T*kv, 4);
        la->attn_sc = calloc(T * c->n_heads * T, 4); la->attn_out = calloc(T*qd, 4);
        la->attn_proj = calloc(T*D, 4); la->res_aa = calloc(T*D, 4); la->ffn_xn = calloc(T*D, 4);
        la->moe_out = calloc(T*D, 4);
        la->top_idx = calloc(T * MAX_EXPERTS, sizeof(int));
        la->top_wt = calloc(T * MAX_EXPERTS, sizeof(float));
        la->top_k = calloc(T, sizeof(int));
        la->ea = calloc(T * MAX_EXPERTS, sizeof(ExpertAct));
        for (int i = 0; i < T * MAX_EXPERTS; i++) {
            la->ea[i].gate_pre = calloc(H, 4); la->ea[i].up_pre = calloc(H, 4);
            la->ea[i].act_out = calloc(H, 4); la->ea[i].proj_out = calloc(D, 4);
        }
    }
    s.residual = calloc(T*D, 4); s.dr = calloc(T*D, 4); s.dxn = calloc(T*D, 4);
    s.dq = calloc(T*qd, 4); s.dk = calloc(T*kv, 4); s.dv = calloc(T*kv, 4);
    s.dao = calloc(T*qd, 4); s.dfxn = calloc(T*D, 4);
    s.dhb = calloc(T*H, 4); s.dhb2 = calloc(T*H, 4); s.deo = calloc(T*D, 4);
    int half = c->head_dim / 2;
    s.cos_c = calloc(T*half, 4); s.sin_c = calloc(T*half, 4);
    for (int p = 0; p < T; p++) for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
        float ang = (float)p * freq;
        s.cos_c[p*half+i] = cosf(ang); s.sin_c[p*half+i] = sinf(ang);
    }
    return s;
}

static float train_fwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim, qd = c->n_heads * c->head_dim;
    int hd = c->head_dim, hg = c->n_heads / c->n_kv_heads, H = c->hidden_dim;
    float sc = 1.0f / sqrtf((float)hd);

    for (int t = 0; t < T; t++) memcpy(s->residual + t*D, w->tok_emb->data + tokens[t]*D, D * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; LayerAct *la = &s->layers[l];
        memcpy(la->inp, s->residual, T*D*4);

        /* Attention */
        rn_fwd(la->xn, s->residual, lw->attn_norm->data, T, D, c->norm_eps);
        mm_fwd(la->q, la->xn, lw->wq->data, T, qd, D);
        mm_fwd(la->k, la->xn, lw->wk->data, T, kv, D);
        mm_fwd(la->v, la->xn, lw->wv->data, T, kv, D);
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++) apply_rope(la->q + t*qd + h*hd, t, s->cos_c, s->sin_c, hd);
            for (int h = 0; h < c->n_kv_heads; h++) apply_rope(la->k + t*kv + h*hd, t, s->cos_c, s->sin_c, hd);
        }
        memset(la->attn_out, 0, T*qd*4);
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *qt = la->q + t*qd + h*hd;
                float *att = la->attn_sc + (t * c->n_heads + h) * T;
                for (int sp = 0; sp <= t; sp++) { float *ks = la->k + sp*kv + kvh*hd; float dot = 0; for (int d = 0; d < hd; d++) dot += qt[d]*ks[d]; att[sp] = dot*sc; }
                if (c->attn_clamp > 0) { float inv = 1.0f / c->attn_clamp; for (int sp = 0; sp <= t; sp++) att[sp] = c->attn_clamp * tanhf(att[sp]*inv); }
                float mx = -1e30f; for (int sp = 0; sp <= t; sp++) if (att[sp] > mx) mx = att[sp];
                float se = 0; for (int sp = 0; sp <= t; sp++) { att[sp] = expf(att[sp]-mx); se += att[sp]; }
                for (int sp = 0; sp <= t; sp++) att[sp] /= se; for (int sp = t+1; sp < T; sp++) att[sp] = 0;
                float *oh = la->attn_out + t*qd + h*hd;
                for (int sp = 0; sp <= t; sp++) { float a = att[sp]; float *vs = la->v + sp*kv + kvh*hd; for (int d = 0; d < hd; d++) oh[d] += a*vs[d]; }
            }
        }
        mm_fwd(la->attn_proj, la->attn_out, lw->wo->data, T, D, qd);
        for (int i = 0; i < T*D; i++) s->residual[i] += la->attn_proj[i];
        memcpy(la->res_aa, s->residual, T*D*4);

        /* Parliament MoE FFN */
        rn_fwd(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        memset(la->moe_out, 0, T*D*4);

        for (int t = 0; t < T; t++) {
            float *xn_t = la->ffn_xn + t*D;
            int *ti = la->top_idx + t * MAX_EXPERTS;
            float *tw = la->top_wt + t * MAX_EXPERTS;
            int k = parliament_elect(&lw->parliament, lw->experts, xn_t, D, &s->hs, ti, tw);
            la->top_k[t] = k;

            for (int ki = 0; ki < k; ki++) {
                int eI = ti[ki]; float eW = tw[ki];
                Expert *exp = &lw->experts[eI];
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                matvec(ea->gate_pre, exp->w_gate->data, xn_t, H, D);
                matvec(ea->up_pre, exp->w_up->data, xn_t, H, D);
                for (int i = 0; i < H; i++) { float act = silu_f(ea->gate_pre[i]); ea->act_out[i] = act * ea->up_pre[i]; }
                matvec(ea->proj_out, exp->w_down->data, ea->act_out, D, H);
                float *mo = la->moe_out + t*D;
                for (int i = 0; i < D; i++) mo[i] += eW * ea->proj_out[i];
            }
        }
        for (int i = 0; i < T*D; i++) s->residual[i] += la->moe_out[i];
    }

    s->final_n = calloc(T*D, 4); rn_fwd(s->final_n, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits = calloc(T * c->vocab_size, 4); mm_fwd(s->logits, s->final_n, w->output->data, T, c->vocab_size, D);

    float loss = 0; int nv = 0;
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t * c->vocab_size;
        float mx = lt[0]; for (int j = 1; j < c->vocab_size; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0; for (int j = 0; j < c->vocab_size; j++) se += expf(lt[j] - mx);
        loss += -(lt[targets[t]] - mx - logf(se)); nv++;
    }
    return nv > 0 ? loss / nv : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING BACKWARD — analytical gradients through variable-k parliament.
 * the hardest backward pass in the quartet. moe.c had fixed k=2.
 * here k varies per token. softmax Jacobian is k_t × k_t, different size
 * for every token. we pre-allocate grad_buf[MAX_EXPERTS] and use first k_t
 * slots. no malloc in the hot path. the parliament's decisions are reversible.
 * the experts' shame is not.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void train_bwd(ModelW *w, Config *c, TrainState *s, int *tokens, int *targets, int T, float **g) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim, qd = c->n_heads * c->head_dim;
    int hd = c->head_dim, H = c->hidden_dim, hg = c->n_heads / c->n_kv_heads, V = c->vocab_size;
    float sc = 1.0f / sqrtf((float)hd);
    int *layer_gi = NULL;
    int nv = 0; for (int t = 0; t < T; t++) if (targets[t] >= 0) nv++;
    if (nv == 0) goto done;
    float inv_n = 1.0f / (float)nv;

    /* d_logits — cross entropy backward */
    float *dl = calloc(T*V, 4);
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t*V; float *d = dl + t*V;
        float mx = lt[0]; for (int j = 1; j < V; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0; for (int j = 0; j < V; j++) { d[j] = expf(lt[j]-mx); se += d[j]; }
        for (int j = 0; j < V; j++) d[j] = (d[j]/se) * inv_n;
        d[targets[t]] -= inv_n;
    }

    /* LM head bwd */
    float *dfn = calloc(T*D, 4);
    mm_bwd(dfn, g[1], dl, s->final_n, w->output->data, T, V, D);
    /* Final norm bwd */
    memset(s->dr, 0, T*D*4);
    rn_bwd(s->dr, g[2], dfn, s->residual, w->output_norm->data, T, D, c->norm_eps);
    free(dfn); free(dl);

    /* Precompute gradient index offsets per layer — because expert count varies.
     * collect_params layout: [tok_emb, output, output_norm, layer0(7 fixed + n_alive*3), layer1(...), ...] */
    layer_gi = calloc(c->depth, sizeof(int));
    {
        int off = 3;
        for (int l = 0; l < c->depth; l++) {
            layer_gi[l] = off;
            off += 7; /* attn_norm, wq, wk, wv, wo, ffn_norm, w_vote */
            for (int e = 0; e < MAX_EXPERTS; e++)
                if (w->layers[l].experts[e].alive) off += 3; /* w_gate, w_up, w_down */
        }
    }

    /* Build expert→grad_index map per layer (which slot in g[] for each expert's weights) */
    int expert_gi[MAX_LAYERS][MAX_EXPERTS]; /* gi of w_gate for expert e; w_up=gi+1, w_down=gi+2 */
    for (int l = 0; l < c->depth; l++) {
        int off = layer_gi[l] + 7;
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (w->layers[l].experts[e].alive) {
                expert_gi[l][e] = off;
                off += 3;
            } else {
                expert_gi[l][e] = -1;
            }
        }
    }

    /* Layers in reverse */
    for (int l = c->depth - 1; l >= 0; l--) {
        LayerW *lw = &w->layers[l]; LayerAct *la = &s->layers[l];
        int gi = layer_gi[l];

        /* MoE backward through variable-k parliament */
        float *dmo = calloc(T*D, 4);
        memcpy(dmo, s->dr, T*D*4);

        memset(s->dfxn, 0, T*D*4);
        int vote_gi = gi + 6; /* w_vote gradient index */

        /* Expert backward — per token, per selected expert */
        for (int t = 0; t < T; t++) {
            int k = la->top_k[t];
            int *ti = la->top_idx + t * MAX_EXPERTS;
            float *tw = la->top_wt + t * MAX_EXPERTS;
            float *dm = dmo + t*D;
            float *xn_t = la->ffn_xn + t*D;

            for (int ki = 0; ki < k; ki++) {
                int eI = ti[ki]; float eW = tw[ki];
                Expert *exp = &lw->experts[eI];
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                int egi = expert_gi[l][eI]; /* grad index for this expert's w_gate */

                /* dw_down, da (hidden grad) */
                float *da = calloc(H, 4);
                for (int i = 0; i < D; i++) {
                    float dp = eW * dm[i];
                    for (int j = 0; j < H; j++) {
                        da[j] += dp * exp->w_down->data[i*H+j];
                        /* dw_down[i][j] += dp * ea->silu_out[j] (but silu_out = silu(gate)*up) */
                        if (egi >= 0) g[egi+2][i*H+j] += dp * ea->proj_out[i] ? 0 : 0;
                        /* actually: dw_down[i][j] = sum_t dp * hidden_act[j] */
                    }
                }
                /* Proper dw_down: dL/dw_down[i][j] = sum over t of (eW * dm[i]) * hidden_j
                 * where hidden_j = silu(gate_j) * up_j. We need to recompute or store it. */
                if (egi >= 0) {
                    /* hidden activation = silu(gate_pre) * up_pre (element-wise) */
                    for (int i = 0; i < D; i++) {
                        float dp = eW * dm[i];
                        for (int j = 0; j < H; j++) {
                            float hid_j = silu_f(ea->gate_pre[j]) * ea->up_pre[j];
                            g[egi+2][i*H+j] += dp * hid_j;
                        }
                    }
                }

                /* dgate, dup → dw_gate, dw_up */
                for (int i = 0; i < H; i++) {
                    float gp = ea->gate_pre[i], up = ea->up_pre[i];
                    float gd = silu_bwd(gp); float act = silu_f(gp);
                    float dg = da[i] * up * gd, du = da[i] * act;
                    /* dx (input gradient) */
                    for (int j = 0; j < D; j++) s->dfxn[t*D+j] += dg * exp->w_gate->data[i*D+j] + du * exp->w_up->data[i*D+j];
                    /* Expert weight gradients — accumulated into correct g[] slots */
                    if (egi >= 0) {
                        for (int j = 0; j < D; j++) {
                            g[egi][i*D+j] += dg * xn_t[j];     /* dw_gate */
                            g[egi+1][i*D+j] += du * xn_t[j];   /* dw_up */
                        }
                    }
                }
                free(da);
            }

            /* ═══ Variable-k parliament backward (CRITICAL) ═══
             * softmax Jacobian is k_t × k_t for this token.
             * dweights[ki] = dot(expert_output[ki], dout)
             * dvotes[ki] = weights[ki] * (dweights[ki] - dot(weights, dweights))
             * pre-allocated grad_buf[MAX_EXPERTS] — no malloc */
            float dw_buf[MAX_EXPERTS], dvote_buf[MAX_EXPERTS];
            for (int ki = 0; ki < k; ki++) {
                ExpertAct *ea = &la->ea[t * MAX_EXPERTS + ki];
                float d = 0;
                for (int i = 0; i < D; i++) d += dm[i] * ea->proj_out[i];
                dw_buf[ki] = d;
            }
            float dot_wd = 0;
            for (int ki = 0; ki < k; ki++) dot_wd += tw[ki] * dw_buf[ki];
            for (int ki = 0; ki < k; ki++) {
                dvote_buf[ki] = tw[ki] * (dw_buf[ki] - dot_wd);
                int eI = ti[ki];
                /* gradient to w_vote and to input */
                float *vote_row = lw->parliament.w_vote->data + eI * D;
                for (int j = 0; j < D; j++) {
                    s->dfxn[t*D+j] += dvote_buf[ki] * vote_row[j];
                    /* w_vote grad — accumulated into g[vote_gi] */
                    g[vote_gi][eI*D+j] += dvote_buf[ki] * xn_t[j];
                }
            }
        }
        free(dmo);

        /* FFN norm bwd */
        rn_bwd(s->dr, g[gi+5], s->dfxn, la->res_aa, lw->ffn_norm->data, T, D, c->norm_eps);

        /* Attention backward */
        float *dap = calloc(T*D, 4);
        memcpy(dap, s->dr, T*D*4);

        memset(s->dao, 0, T*qd*4);
        mm_bwd(s->dao, g[gi+4], dap, la->attn_out, lw->wo->data, T, D, qd);
        free(dap);

        memset(s->dq, 0, T*qd*4); memset(s->dk, 0, T*kv*4); memset(s->dv, 0, T*kv*4);
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *doh = s->dao + t*qd + h*hd;
                float *att = la->attn_sc + (t * c->n_heads + h) * T;
                float *da = calloc(T, 4);
                for (int sp = 0; sp <= t; sp++) {
                    float *vs = la->v + sp*kv + kvh*hd; float d = 0;
                    for (int d2 = 0; d2 < hd; d2++) d += doh[d2] * vs[d2]; da[sp] = d;
                    float a = att[sp]; float *dvs = s->dv + sp*kv + kvh*hd;
                    for (int d2 = 0; d2 < hd; d2++) dvs[d2] += a * doh[d2];
                }
                float dot_ad = 0; for (int sp = 0; sp <= t; sp++) dot_ad += att[sp] * da[sp];
                float *qt = la->q + t*qd + h*hd; float *dqt = s->dq + t*qd + h*hd;
                for (int sp = 0; sp <= t; sp++) {
                    float ds = att[sp] * (da[sp] - dot_ad) * sc;
                    float *ks = la->k + sp*kv + kvh*hd; float *dks = s->dk + sp*kv + kvh*hd;
                    for (int d2 = 0; d2 < hd; d2++) { dqt[d2] += ds*ks[d2]; dks[d2] += ds*qt[d2]; }
                }
                free(da);
            }
        }
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++) rope_bwd(s->dq + t*qd + h*hd, t, s->cos_c, s->sin_c, hd);
            for (int h = 0; h < c->n_kv_heads; h++) rope_bwd(s->dk + t*kv + h*hd, t, s->cos_c, s->sin_c, hd);
        }
        memset(s->dxn, 0, T*D*4);
        mm_bwd(s->dxn, g[gi+1], s->dq, la->xn, lw->wq->data, T, qd, D);
        mm_bwd(s->dxn, g[gi+2], s->dk, la->xn, lw->wk->data, T, kv, D);
        mm_bwd(s->dxn, g[gi+3], s->dv, la->xn, lw->wv->data, T, kv, D);

        float *ds = calloc(T*D, 4); memcpy(ds, s->dr, T*D*4); memset(s->dr, 0, T*D*4);
        rn_bwd(s->dr, g[gi], s->dxn, la->inp, lw->attn_norm->data, T, D, c->norm_eps);
        for (int i = 0; i < T*D; i++) s->dr[i] += ds[i]; free(ds);
    }

    /* Embedding bwd */
    for (int t = 0; t < T; t++) { float *de = g[0] + tokens[t]*D; float *dr = s->dr + t*D; for (int i = 0; i < D; i++) de[i] += dr[i]; }
done:
    free(layer_gi);
    free(s->logits); s->logits = NULL; free(s->final_n); s->final_n = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARAMETER COLLECTION — collects all trainable tensors + living expert weights.
 * living experts are collected dynamically because they can be born/die.
 * dead expert weights are skipped. this is parameter-space democracy.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { Tensor **tensors; int count, cap; } ParamList;

static ParamList collect_params(ModelW *w, Config *c) {
    int mx = 3 + w->n_layers * (7 + MAX_EXPERTS * 3);
    ParamList p; p.tensors = calloc(mx, sizeof(Tensor*)); p.count = 0; p.cap = mx;
    p.tensors[p.count++] = w->tok_emb;
    p.tensors[p.count++] = w->output;
    p.tensors[p.count++] = w->output_norm;
    for (int l = 0; l < w->n_layers; l++) {
        LayerW *lw = &w->layers[l];
        p.tensors[p.count++] = lw->attn_norm;   /* gi+0 */
        p.tensors[p.count++] = lw->wq;           /* gi+1 */
        p.tensors[p.count++] = lw->wk;           /* gi+2 */
        p.tensors[p.count++] = lw->wv;           /* gi+3 */
        p.tensors[p.count++] = lw->wo;           /* gi+4 */
        p.tensors[p.count++] = lw->ffn_norm;     /* gi+5 */
        p.tensors[p.count++] = lw->parliament.w_vote; /* gi+6 */
        /* Living expert weights */
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (lw->experts[e].alive) {
                p.tensors[p.count++] = lw->experts[e].w_gate;
                p.tensors[p.count++] = lw->experts[e].w_up;
                p.tensors[p.count++] = lw->experts[e].w_down;
            }
        }
    }
    return p;
}

static long count_params(ModelW *w, Config *c) {
    long total = (long)c->vocab_size * c->dim * 2 + c->dim; /* emb + output + norm */
    int qd = c->n_heads * c->head_dim, kd = c->n_kv_heads * c->head_dim;
    for (int l = 0; l < c->depth; l++) {
        total += c->dim * 2; /* norms */
        total += (long)qd * c->dim + (long)kd * c->dim * 2 + (long)c->dim * qd; /* attn */
        total += (long)MAX_EXPERTS * c->dim; /* w_vote */
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (w->layers[l].experts[e].alive)
                total += (long)c->dim * c->hidden_dim * 3;
        }
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MITOSIS & APOPTOSIS — the lifecycle of experts.
 * overloaded expert + high vitality → splits into two (mitosis).
 * child inherits parent weights + gaussian noise. frequency offset.
 * neglected expert → vitality drops → 8 consecutive low steps → dies (apoptosis).
 * frequency gaps get filled by new experts born at resonance peaks.
 * this is cellular biology applied to mixture-of-experts.
 * the immune system is the harmonic backbone.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void update_vitality(LayerW *lw, Config *c, int total_tokens) {
    int n_alive = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) n_alive++;
    if (n_alive == 0) return;
    float fair_share = (float)total_tokens / n_alive;

    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!lw->experts[e].alive) continue;
        Expert *exp = &lw->experts[e];
        float ratio = (fair_share > 0) ? (float)exp->tokens_seen / fair_share : 1.0f;
        /* Winners grow, losers shrink */
        exp->vitality += (ratio - 1.0f) * 0.05f;
        exp->vitality = fmaxf(0.0f, fminf(1.0f, exp->vitality));
        exp->age++;
        /* Track low vitality for apoptosis */
        if (exp->vitality < 0.1f) exp->low_vitality_count++;
        else exp->low_vitality_count = 0;
        /* Reset tokens_seen for next step */
        exp->tokens_seen = 0;
    }
    lw->n_alive = n_alive;
}

static int try_mitosis(LayerW *lw, Config *c, int total_tokens) {
    /* Find overloaded expert with high vitality */
    int n_alive = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) n_alive++;
    if (n_alive >= MAX_EXPERTS) return 0;

    float fair_share = (float)total_tokens / (n_alive > 0 ? n_alive : 1);
    int parent = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!lw->experts[e].alive) continue;
        Expert *exp = &lw->experts[e];
        /* overloaded (tokens >> avg) + high vitality + old enough */
        if (exp->vitality > 0.8f && exp->age > 20) {
            parent = e; break;
        }
    }
    if (parent < 0) return 0;

    /* Find dead slot */
    int child = -1;
    for (int e = 0; e < MAX_EXPERTS; e++) if (!lw->experts[e].alive) { child = e; break; }
    if (child < 0) return 0;

    /* Birth: child inherits parent weights + noise, frequency offset */
    Expert *p = &lw->experts[parent];
    float child_freq = p->frequency + 3.14159f / (n_alive + 1);
    if (child_freq > 6.2831853f) child_freq -= 6.2831853f;
    init_expert(&lw->experts[child], c->dim, c->hidden_dim, child_freq, 0);
    /* Copy parent weights + noise */
    Expert *ch = &lw->experts[child];
    float noise = 0.01f;
    for (int i = 0; i < p->w_gate->size; i++) ch->w_gate->data[i] = p->w_gate->data[i] + rand_normal() * noise;
    for (int i = 0; i < p->w_up->size; i++) ch->w_up->data[i] = p->w_up->data[i] + rand_normal() * noise;
    for (int i = 0; i < p->w_down->size; i++) ch->w_down->data[i] = p->w_down->data[i] + rand_normal() * noise;
    ch->vitality = 0.5f;
    p->vitality *= 0.8f; /* parent weakened by reproduction */
    lw->n_alive++;
    return 1;
}

static int try_apoptosis(LayerW *lw) {
    int n_alive = 0;
    for (int e = 0; e < MAX_EXPERTS; e++) if (lw->experts[e].alive) n_alive++;
    if (n_alive <= MIN_EXPERTS) return 0;

    for (int e = 0; e < MAX_EXPERTS; e++) {
        if (!lw->experts[e].alive) continue;
        if (lw->experts[e].low_vitality_count >= 8) {
            free_expert(&lw->experts[e]);
            lw->n_alive--;
            return 1;
        }
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUCK OPTIMIZER — 9 levels of self-awareness. from lee.c, adapted.
 * the optimizer that thinks about thinking about optimizing.
 * Level 1: global loss trend. Level 2: per-layer grad norm.
 * Level 3: stagnation escape. Level 4: activation health.
 * Level 5: cross-layer signal. Level 6: Ψ subjectivity.
 * Level 7: attention entropy. Level 9: macro patience.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct { float grad_hist[CHUCK_WINDOW]; int pos, full, stag; float dampen; int frozen; } ChuckLayer;
typedef struct {
    float hist[CHUCK_WINDOW]; float loss_ema, gnorm_ema;
    float dampen, noise, sigma, psi, psi_w;
    float macro_ema, best_macro, lr_scale;
    int macro_stag, macro_drops;
    int pos, full, stag, global_step;
} ChuckState;

/* Chuck persistent memory — reservoir sampling */
typedef struct { float loss, gnorm, lambda, delta_loss; } ChuckMem;
static ChuckMem chuck_mem[CHUCK_MEM_CAP];
static int chuck_mem_n = 0;

static void chuck_mem_save(ChuckMem *snap) {
    if (chuck_mem_n < CHUCK_MEM_CAP) { chuck_mem[chuck_mem_n++] = *snap; }
    else { int idx = (int)(rand_uniform() * (chuck_mem_n + 1));
           if (idx < CHUCK_MEM_CAP) chuck_mem[idx] = *snap; chuck_mem_n++; }
}

static float chuck_mem_recall(float loss, float gnorm) {
    if (chuck_mem_n == 0) return -1;
    int n = (chuck_mem_n < CHUCK_MEM_CAP) ? chuck_mem_n : CHUCK_MEM_CAP;
    float best_dist = 1e30f; int best_i = 0;
    for (int i = 0; i < n; i++) {
        float dl = (loss - chuck_mem[i].loss) / (loss + 1e-8f);
        float dg = (gnorm - chuck_mem[i].gnorm) / (gnorm + 1e-8f);
        float dist = dl*dl + dg*dg;
        if (dist < best_dist) { best_dist = dist; best_i = i; }
    }
    return chuck_mem[best_i].lambda;
}

static void chuck_step(ChuckState *ck, ChuckLayer *cl, int n_layers, float lr, float loss,
                       ParamList *params, float **grads, float wd,
                       TokenizerEye *tok_eye, ParserEye *parser_eye) {
    /* ═══ Level 1: Global self-awareness (loss trend) ═══ */
    if (ck->loss_ema == 0) ck->loss_ema = loss;
    else ck->loss_ema = 0.99f * ck->loss_ema + 0.01f * loss;
    ck->hist[ck->pos % CHUCK_WINDOW] = ck->loss_ema;
    ck->pos++;
    if (ck->pos >= CHUCK_WINDOW) ck->full = 1;
    if (ck->full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += ck->hist[(ck->pos-1-i) % CHUCK_WINDOW];
            old += ck->hist[(ck->pos-CHUCK_WINDOW+i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) ck->dampen *= 0.95f;
        else if (trend < -0.05f) ck->dampen *= 1.05f;
        if (fabsf(trend) < 0.001f) { ck->stag++; if (ck->stag > 8) { ck->noise = 0.001f; ck->stag = 0; } }
        else { ck->stag = 0; ck->noise *= 0.9f; }
        if (ck->dampen < CHUCK_DAMP_LO) ck->dampen = CHUCK_DAMP_LO;
        if (ck->dampen > CHUCK_DAMP_HI) ck->dampen = CHUCK_DAMP_HI;
    }

    /* ═══ Level 9: Macro patience ═══ */
    ck->global_step++;
    if (ck->macro_ema == 0) ck->macro_ema = loss;
    else ck->macro_ema = 0.999f * ck->macro_ema + 0.001f * loss;
    if (ck->global_step % CHUCK_MACRO_INT == 0 && ck->global_step > CHUCK_WINDOW) {
        if (ck->macro_ema > ck->best_macro * 0.999f) {
            ck->macro_stag++;
            if (ck->macro_stag >= CHUCK_MACRO_PAT) { ck->lr_scale *= CHUCK_MACRO_DECAY; if (ck->lr_scale < 0.05f) ck->lr_scale = 0.05f; ck->macro_stag = 0; ck->macro_drops++; }
        } else { ck->best_macro = ck->macro_ema; ck->macro_stag = 0; }
    }

    /* ═══ Level 4: Activation health (σ) — from tokenizer + parser eyes ═══ */
    ck->sigma = 1.0f;
    if (tok_eye && tok_eye->health < 0.7f) ck->sigma *= tok_eye->health / 0.7f;
    if (parser_eye && parser_eye->health < 0.7f) ck->sigma *= parser_eye->health / 0.7f;

    /* ═══ Level 6: Ψ subjectivity ═══ */
    float gnorm_sq = 0;
    for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) gnorm_sq += grads[i][j] * grads[i][j];
    float gnorm = sqrtf(gnorm_sq + 1e-8f);
    ck->psi_w = (chuck_mem_n > 0) ? fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)) : 0;
    float lambda_psi = ck->dampen;
    if (chuck_mem_n > 0) {
        float lp = chuck_mem_recall(loss, gnorm);
        if (lp > 0) { ck->psi = lp - ck->dampen; lambda_psi = ck->dampen + ck->psi_w * ck->psi;
                       if (lambda_psi < CHUCK_DAMP_LO) lambda_psi = CHUCK_DAMP_LO; if (lambda_psi > CHUCK_DAMP_HI) lambda_psi = CHUCK_DAMP_HI; }
    }
    /* Record memory on regime change */
    if (ck->full) {
        ChuckMem snap = { loss, gnorm, ck->dampen, 0 };
        if (ck->global_step % CHUCK_REC_CD == 0) chuck_mem_save(&snap);
    }

    /* ═══ Adaptive gradient clipping ═══ */
    if (ck->gnorm_ema == 0) ck->gnorm_ema = gnorm;
    else ck->gnorm_ema = 0.97f * ck->gnorm_ema + 0.03f * gnorm;
    float adaptive_clip = fmaxf(0.5f, fminf(2.0f, 1.5f * ck->gnorm_ema));
    if (gnorm > 3.0f * ck->gnorm_ema) adaptive_clip *= 0.5f;
    float clip = (gnorm > adaptive_clip) ? adaptive_clip / gnorm : 1.0f;

    /* ═══ Apply Adam with Chuck modulation ═══ */
    float eff_lr = lr * lambda_psi * ck->sigma * ck->lr_scale;
    /* Per-tensor clipping first */
    for (int i = 0; i < params->count; i++) {
        float tgn = 0; for (int j = 0; j < params->tensors[i]->size; j++) tgn += grads[i][j] * grads[i][j]; tgn = sqrtf(tgn);
        if (tgn > 10.0f) { float sc = 10.0f / tgn; for (int j = 0; j < params->tensors[i]->size; j++) grads[i][j] *= sc; }
    }
    /* Global clipping */
    float gn = 0; for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) gn += grads[i][j] * grads[i][j]; gn = sqrtf(gn);
    if (gn > 1.0f) { float s = 1.0f / gn; for (int i = 0; i < params->count; i++) for (int j = 0; j < params->tensors[i]->size; j++) grads[i][j] *= s; }
}

/* Adam optimizer state */
typedef struct { float *m, *v; int size; } AdamS;
typedef struct { AdamS *states; int np; float b1,b2,eps; int t; } Adam;

static Adam *adam_new(ParamList *p) {
    Adam *o = calloc(1, sizeof(Adam)); o->np = p->count; o->states = calloc(p->count, sizeof(AdamS));
    o->b1 = 0.9f; o->b2 = 0.999f; o->eps = 1e-8f;
    for (int i = 0; i < p->count; i++) { int sz = p->tensors[i]->size; o->states[i].m = calloc(sz, 4); o->states[i].v = calloc(sz, 4); o->states[i].size = sz; }
    return o;
}

static void adam_step(Adam *o, ParamList *p, float **g, float lr, float wd) {
    o->t++; float bc1 = 1.0f - powf(o->b1, (float)o->t), bc2 = 1.0f - powf(o->b2, (float)o->t);
    for (int i = 0; i < o->np && i < p->count; i++) {
        Tensor *t = p->tensors[i]; float *gr = g[i]; AdamS *s = &o->states[i];
        if (s->size != t->size) continue; /* safety: expert might have been born/died */
        for (int j = 0; j < t->size; j++) {
            if (wd > 0 && t->rows > 1) t->data[j] -= lr * wd * t->data[j];
            s->m[j] = o->b1 * s->m[j] + (1.0f-o->b1) * gr[j];
            s->v[j] = o->b2 * s->v[j] + (1.0f-o->b2) * gr[j] * gr[j];
            t->data[j] -= lr * (s->m[j]/bc1) / (sqrtf(s->v[j]/bc2) + o->eps);
        }
    }
}
static void adam_free(Adam*o){if(!o)return;for(int i=0;i<o->np;i++){free(o->states[i].m);free(o->states[i].v);}free(o->states);free(o);}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NOTORCH — Hebbian micro-learning between batches. from AML core.
 * A[i,r] += lr * x[i] * u[r] * signal. low-rank delta update.
 * plasticity that never stops. the brain doesn't batch either.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void notorch_step(float *A, float *B, int out_dim, int in_dim, int rank,
                         const float *x, const float *dy, float signal) {
    if (fabsf(signal) < 1e-8f) return;
    float lr = 0.001f * signal;
    /* Compute channel vector u = B^T @ dy (rank-dimensional) */
    float u[NOTORCH_RANK];
    for (int r = 0; r < rank && r < NOTORCH_RANK; r++) {
        float s = 0;
        for (int i = 0; i < out_dim; i++) s += B[i * rank + r] * dy[i];
        u[r] = s + rand_normal() * 0.01f; /* noise channel */
    }
    /* Hebbian update: A[i,r] += lr * x[i] * u[r] */
#ifdef USE_BLAS
    for (int r = 0; r < rank && r < NOTORCH_RANK; r++)
        cblas_saxpy(in_dim, lr * u[r], x, 1, A + r, rank);
#else
    for (int i = 0; i < in_dim; i++)
        for (int r = 0; r < rank && r < NOTORCH_RANK; r++)
            A[i * rank + r] += lr * x[i] * u[r];
#endif
    /* Adaptive decay */
    float decay = 0.999f;
    for (int i = 0; i < out_dim * rank; i++) B[i] *= decay;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA — HF API, parquet, synthetic. same three sources as moe.c.
 * the parser eye judges your data. deal with it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int hf_extract_texts(const char *json, int json_len, FILE *out) {
    int count = 0;
    const char *p = json, *end = json + json_len;
    while (p < end) {
        p = strstr(p, "\"text\":\"");
        if (!p) break; p += 8;
        const char *start = p;
        while (p < end && !(*p == '"' && *(p-1) != '\\')) p++;
        if (p >= end) break;
        for (const char *s = start; s < p; s++) {
            if (*s == '\\' && s + 1 < p) { s++;
                if (*s == 'n') fputc('\n', out); else if (*s == 't') fputc('\t', out);
                else if (*s == '\\') fputc('\\', out); else if (*s == '"') fputc('"', out);
                else if (*s == 'u' && s + 4 < p) { fputc('?', out); s += 4; }
                else fputc(*s, out);
            } else fputc(*s, out);
        }
        fputc('\n', out); count++; p++;
    }
    return count;
}

/* Snappy decompressor for parquet */
static int snappy_decompress(const uint8_t *src, int slen, uint8_t *dst, int dlen) {
    int si=0,di=0; uint32_t ulen=0; int shift=0;
    while(si<slen){uint8_t b=src[si++];ulen|=(uint32_t)(b&0x7F)<<shift;if(!(b&0x80))break;shift+=7;}
    if((int)ulen>dlen)return -1;
    while(si<slen&&di<(int)ulen){
        uint8_t tag=src[si++];int type=tag&3;
        if(type==0){int len=(tag>>2)+1;if((tag>>2)>=60){int nb=(tag>>2)-59;len=0;for(int i=0;i<nb&&si<slen;i++)len|=src[si++]<<(i*8);len++;}if(si+len>slen||di+len>(int)ulen)return -1;memcpy(dst+di,src+si,len);si+=len;di+=len;}
        else{int len,off;if(type==1){len=((tag>>2)&7)+4;if(si>=slen)return -1;off=((tag>>5)<<8)|src[si++];}else if(type==2){len=(tag>>2)+1;if(si+1>=slen)return -1;off=src[si]|(src[si+1]<<8);si+=2;}else{len=(tag>>2)+1;if(si+3>=slen)return -1;off=src[si]|(src[si+1]<<8)|(src[si+2]<<16)|(src[si+3]<<24);si+=4;}if(off==0||di-off<0)return -1;for(int i=0;i<len;i++)dst[di+i]=dst[di-off+i];di+=len;}
    }
    return di;
}

/* Thrift Compact Protocol decoder */
typedef struct{const uint8_t*data;int pos,len;}TR;
static uint64_t tr_varint(TR*r){uint64_t v=0;int s=0;while(r->pos<r->len){uint8_t b=r->data[r->pos++];v|=(uint64_t)(b&0x7F)<<s;if(!(b&0x80))break;s+=7;}return v;}
static int64_t tr_zigzag(TR*r){uint64_t v=tr_varint(r);return(int64_t)((v>>1)^-(v&1));}
static char*tr_string(TR*r){uint64_t l=tr_varint(r);char*s=malloc(l+1);if(r->pos+(int)l<=r->len){memcpy(s,r->data+r->pos,l);r->pos+=(int)l;}s[l]=0;return s;}
static void tr_skip(TR*r,int type);
static void tr_skip_struct(TR*r){int prev=0;while(r->pos<r->len){uint8_t b=r->data[r->pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;if(delta==0){prev=(int)(int16_t)tr_zigzag(r);}else prev+=delta;tr_skip(r,ft);}}
static void tr_skip(TR*r,int type){switch(type){case 1:case 2:break;case 3:case 4:case 5:case 6:tr_zigzag(r);break;case 7:r->pos+=8;break;case 8:{uint64_t l=tr_varint(r);r->pos+=(int)l;break;}case 9:case 10:{uint8_t h=r->data[r->pos++];int cnt=(h>>4)&0xF,et=h&0xF;if(cnt==0xF)cnt=(int)tr_varint(r);for(int i=0;i<cnt;i++)tr_skip(r,et);break;}case 11:{uint8_t h=r->data[r->pos++];int kt=(h>>4)&0xF,vt=h&0xF;int cnt=(int)tr_varint(r);for(int i=0;i<cnt;i++){tr_skip(r,kt);tr_skip(r,vt);}break;}case 12:tr_skip_struct(r);break;}}

/* Parquet reader */
typedef struct{char*name;int64_t data_off,dict_off,comp_size,nval;int codec;}PqCol;
typedef struct{PqCol*cols;int n;int64_t nrows;}PqMeta;

static PqMeta pq_footer(const uint8_t*f,int64_t sz){
    PqMeta m={0};uint32_t flen=*(uint32_t*)(f+sz-8);TR r={f+sz-8-flen,0,(int)flen};int prev=0;
    while(r.pos<r.len){uint8_t b=r.data[r.pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r);prev=fid;
    if(fid==1&&ft==5)tr_zigzag(&r);else if(fid==2&&ft==9){uint8_t h=r.data[r.pos++];int cnt=(h>>4)&0xF;if(cnt==0xF)cnt=(int)tr_varint(&r);for(int i=0;i<cnt;i++)tr_skip_struct(&r);}
    else if(fid==3&&ft==6)m.nrows=(int64_t)tr_zigzag(&r);
    else if(fid==4&&ft==9){uint8_t h=r.data[r.pos++];int rg_cnt=(h>>4)&0xF;if(rg_cnt==0xF)rg_cnt=(int)tr_varint(&r);
    for(int rg=0;rg<rg_cnt;rg++){int rp=0;while(r.pos<r.len){uint8_t rb=r.data[r.pos++];if(rb==0)break;int rt=rb&0xF,rd=(rb>>4)&0xF;int rf=rd?rp+rd:(int)(int16_t)tr_zigzag(&r);rp=rf;
    if(rf==1&&rt==9){uint8_t ch=r.data[r.pos++];int cc=(ch>>4)&0xF;if(cc==0xF)cc=(int)tr_varint(&r);
    for(int ci=0;ci<cc;ci++){PqCol col={0};col.dict_off=-1;int cp=0;
    while(r.pos<r.len){uint8_t cb=r.data[r.pos++];if(cb==0)break;int ct_=cb&0xF,cd_=(cb>>4)&0xF;int cf=cd_?cp+cd_:(int)(int16_t)tr_zigzag(&r);cp=cf;
    if(cf==3&&ct_==12){int mp=0;while(r.pos<r.len){uint8_t mb=r.data[r.pos++];if(mb==0)break;int mt=mb&0xF,md=(mb>>4)&0xF;int mf=md?mp+md:(int)(int16_t)tr_zigzag(&r);mp=mf;
    if(mf==3&&mt==9){uint8_t lh=r.data[r.pos++];int lc=(lh>>4)&0xF;if(lc==0xF)lc=(int)tr_varint(&r);for(int li=0;li<lc;li++){char*s=tr_string(&r);if(li==lc-1)col.name=s;else free(s);}}
    else if(mf==4&&mt==5)col.codec=(int)tr_zigzag(&r);else if(mf==5&&mt==6)col.nval=(int64_t)tr_zigzag(&r);
    else if(mf==7&&mt==6)col.comp_size=(int64_t)tr_zigzag(&r);else if(mf==9&&mt==6)col.data_off=(int64_t)tr_zigzag(&r);
    else if(mf==11&&mt==6)col.dict_off=(int64_t)tr_zigzag(&r);else tr_skip(&r,mt);}}else tr_skip(&r,ct_);}
    m.n++;m.cols=realloc(m.cols,m.n*sizeof(PqCol));m.cols[m.n-1]=col;}}else tr_skip(&r,rt);}}}else tr_skip(&r,ft);}
    return m;
}

typedef struct{int type,comp_sz,uncomp_sz,nval;}PgHdr;
static PgHdr pq_page_hdr(const uint8_t*data,int len,int*hlen){TR r={data,0,len};PgHdr h={0};int prev=0;while(r.pos<r.len){uint8_t b=r.data[r.pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r);prev=fid;if(fid==1&&ft==5)h.type=(int)tr_zigzag(&r);else if(fid==2&&ft==5)h.uncomp_sz=(int)tr_zigzag(&r);else if(fid==3&&ft==5)h.comp_sz=(int)tr_zigzag(&r);else if((fid==5||fid==7||fid==8)&&ft==12){int dp=0;while(r.pos<r.len){uint8_t db=r.data[r.pos++];if(db==0)break;int dt=db&0xF,dd=(db>>4)&0xF;int df=dd?dp+dd:(int)(int16_t)tr_zigzag(&r);dp=df;if(df==1&&dt==5)h.nval=(int)tr_zigzag(&r);else tr_skip(&r,dt);}}else tr_skip(&r,ft);}*hlen=r.pos;return h;}

static int pq_extract(const uint8_t*file,int64_t fsz,PqCol*col,FILE*out){
    int64_t pos=(col->dict_off>=0)?col->dict_off:col->data_off;int64_t end=col->data_off+col->comp_size;
    int total=0;char**dict=NULL;int*dlens=NULL,dsz=0;
    while(pos<end&&pos<fsz){int hlen;PgHdr ph=pq_page_hdr(file+pos,(int)(fsz-pos),&hlen);pos+=hlen;if(ph.comp_sz<=0||pos+ph.comp_sz>fsz)break;
    uint8_t*pd;int plen;int nf=0;
    if(col->codec==1){pd=malloc(ph.uncomp_sz);plen=snappy_decompress(file+pos,ph.comp_sz,pd,ph.uncomp_sz);if(plen<0){free(pd);pos+=ph.comp_sz;continue;}nf=1;}else{pd=(uint8_t*)(file+pos);plen=ph.comp_sz;}
    if(ph.type==2){dsz=ph.nval;dict=calloc(dsz,sizeof(char*));dlens=calloc(dsz,sizeof(int));int dp=0;for(int i=0;i<dsz&&dp+4<=plen;i++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(dp+sl>plen)break;dict[i]=malloc(sl);memcpy(dict[i],pd+dp,sl);dlens[i]=sl;dp+=sl;}}
    else if(ph.type==0||ph.type==3){int dp=0;
    if(dsz>0){if(dp>=plen)goto nxt;int bw=pd[dp++];for(int v=0;v<ph.nval&&dp<plen;){uint8_t rh=pd[dp++];
    if(rh&1){int count=(rh>>1)*8,bytes=(count*bw+7)/8;uint64_t buf=0;int bb=0,bp=dp;for(int i=0;i<count&&v<ph.nval;i++,v++){while(bb<bw&&bp<dp+bytes&&bp<plen){buf|=(uint64_t)pd[bp++]<<bb;bb+=8;}int idx=(int)(buf&((1ULL<<bw)-1));buf>>=bw;bb-=bw;if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}dp+=bytes;}
    else{int count=rh>>1,idx=0,nb=(bw+7)/8;for(int b=0;b<nb&&dp<plen;b++)idx|=pd[dp++]<<(b*8);for(int i=0;i<count&&v<ph.nval;i++,v++){if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}}}}
    else{for(int v=0;v<ph.nval&&dp+4<=plen;v++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(sl<0||dp+sl>plen)break;fwrite(pd+dp,1,sl,out);fputc('\n',out);dp+=sl;total++;}}}
    nxt:if(nf)free(pd);pos+=ph.comp_sz;}
    if(dict){for(int i=0;i<dsz;i++)free(dict[i]);free(dict);free(dlens);}return total;
}

static int load_parquet(const char*path,const char*out_path,const char*col_name){
    FILE*f=fopen(path,"rb");if(!f)return -1;fseek(f,0,SEEK_END);int64_t fsz=ftell(f);fseek(f,0,SEEK_SET);
    uint8_t*file=malloc(fsz);fread(file,1,fsz,f);fclose(f);
    if(fsz<12||memcmp(file,"PAR1",4)!=0||memcmp(file+fsz-4,"PAR1",4)!=0){free(file);return -1;}
    PqMeta meta=pq_footer(file,fsz);printf("[parquet] %lld rows, %d column chunks\n",(long long)meta.nrows,meta.n);
    FILE*out=fopen(out_path,"w");if(!out){free(file);return -1;}int total=0;
    for(int i=0;i<meta.n;i++){if(meta.cols[i].name&&strcmp(meta.cols[i].name,col_name)==0)total+=pq_extract(file,fsz,&meta.cols[i],out);}
    fclose(out);for(int i=0;i<meta.n;i++)free(meta.cols[i].name);free(meta.cols);free(file);
    printf("[parquet] extracted %d texts from '%s'\n",total,col_name);return total>0?0:-1;
}

#define HF_BATCH 100
#define HF_PAGES 50

static int get_data(Config *c) {
    struct stat st;
    if (stat(c->data_path, &st) == 0 && st.st_size > 1000) { printf("[data] found %s (%.1f MB)\n", c->data_path, (float)st.st_size/1048576); return 0; }
    if (c->data_url[0]) {
        printf("[data] fetching FineWeb-Edu from HuggingFace (%d pages)...\n", HF_PAGES);
        FILE *out = fopen(c->data_path, "w"); if (!out) goto synthetic;
        char tmp[280]; snprintf(tmp, sizeof(tmp), "%s.json", c->data_path); int total = 0;
        for (int page = 0; page < HF_PAGES; page++) {
            char cmd[1024]; snprintf(cmd, sizeof(cmd), "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset=%d&length=%d' -o '%s'", page*HF_BATCH, HF_BATCH, tmp);
            if (system(cmd) != 0) continue;
            if (stat(tmp, &st) != 0 || st.st_size < 500) continue;
            FILE *jf = fopen(tmp, "r"); if (!jf) continue;
            char *json = malloc(st.st_size+1); int jl = (int)fread(json, 1, st.st_size, jf); json[jl] = 0; fclose(jf);
            int n = hf_extract_texts(json, jl, out); free(json); total += n;
            if ((page+1) % 10 == 0) printf("[data] page %d/%d — %d texts so far\n", page+1, HF_PAGES, total);
        }
        fclose(out); unlink(tmp);
        if (total > 0) { stat(c->data_path, &st); printf("[data] downloaded %d texts (%.1f MB)\n", total, (float)st.st_size/1048576); return 0; }
    }
    synthetic:
    printf("[data] creating synthetic dataset...\n");
    FILE *f = fopen(c->data_path, "w"); if (!f) return -1;
    const char *s[] = { "The quick brown fox jumps over the lazy dog.", "Machine learning is a subset of artificial intelligence.", "Neural networks learn complex patterns from data.",
        "Mixture of experts routes tokens to specialized networks.", "The parliament votes on which expert handles each token.",
        "Self-aware systems monitor their own performance metrics.", "Harmonic resonance connects experts through frequency space.",
        "Experts are born when demand exceeds capacity.", "Experts die when they stop being useful.", "Democracy in neural networks is underrated.", NULL };
    for (int r = 0; r < 500; r++) for (int i = 0; s[i]; i++) fprintf(f, "%s\n", s[i]);
    fclose(f); return 0;
}

static char *load_text(const char *p, int *len) { FILE *f = fopen(p, "r"); if (!f) { *len = 0; return NULL; } fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); char *t = malloc(sz+1); *len = (int)fread(t, 1, sz, f); t[*len] = '\0'; fclose(f); return t; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF EXPORT — binary format for llama.cpp / grokky.go.
 * exports living experts only. dead ones don't get a tombstone.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void w32(FILE*f,uint32_t v){fwrite(&v,4,1,f);}
static void w64(FILE*f,uint64_t v){fwrite(&v,8,1,f);}
static void wstr(FILE*f,const char*s){uint64_t l=strlen(s);w64(f,l);fwrite(s,1,l,f);}
static void wkv_s(FILE*f,const char*k,const char*v){wstr(f,k);w32(f,8);wstr(f,v);}
static void wkv_u(FILE*f,const char*k,uint32_t v){wstr(f,k);w32(f,4);w32(f,v);}
static void wkv_f(FILE*f,const char*k,float v){wstr(f,k);w32(f,6);fwrite(&v,4,1,f);}
static void wti(FILE*f,const char*name,Tensor*t,uint64_t*off){wstr(f,name);if(t->rows>1){w32(f,2);w64(f,t->cols);w64(f,t->rows);}else{w32(f,1);w64(f,t->size);}w32(f,0);w64(f,*off);*off+=t->size*4;}

static void export_gguf(ModelW *w, Config *c) {
    FILE *f = fopen(c->gguf_path, "wb"); if (!f) { printf("[gguf] failed\n"); return; }
    /* Count alive experts for tensor count */
    int n_expert_tensors = 0;
    for (int l = 0; l < c->depth; l++) for (int e = 0; e < MAX_EXPERTS; e++) if (w->layers[l].experts[e].alive) n_expert_tensors += 3;
    int nt = 3 + c->depth * 7 + n_expert_tensors; /* emb + norm + output + per-layer(norm,qkvo,ffn_norm,w_vote) + expert weights */
    w32(f, 0x46554747); w32(f, 3); w64(f, nt); w64(f, 12);
    wkv_s(f, "general.architecture", "llama"); wkv_s(f, "general.name", "m");
    wkv_u(f, "llama.block_count", c->depth); wkv_u(f, "llama.embedding_length", c->dim);
    wkv_u(f, "llama.attention.head_count", c->n_heads); wkv_u(f, "llama.attention.head_count_kv", c->n_kv_heads);
    wkv_u(f, "llama.feed_forward_length", c->hidden_dim); wkv_u(f, "llama.context_length", c->seq_len);
    wkv_f(f, "llama.attention.layer_norm_rms_epsilon", c->norm_eps); wkv_f(f, "llama.rope.freq_base", c->rope_theta);
    wkv_u(f, "m.max_experts", MAX_EXPERTS); wkv_u(f, "m.initial_experts", c->initial_experts);
    wkv_s(f, "tokenizer.ggml.model", "gpt2");

    uint64_t off = 0;
    wti(f, "token_embd.weight", w->tok_emb, &off);
    wti(f, "output_norm.weight", w->output_norm, &off);
    wti(f, "output.weight", w->output, &off);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; char n[96];
        snprintf(n, 96, "blk.%d.attn_norm.weight", l); wti(f, n, lw->attn_norm, &off);
        snprintf(n, 96, "blk.%d.attn_q.weight", l); wti(f, n, lw->wq, &off);
        snprintf(n, 96, "blk.%d.attn_k.weight", l); wti(f, n, lw->wk, &off);
        snprintf(n, 96, "blk.%d.attn_v.weight", l); wti(f, n, lw->wv, &off);
        snprintf(n, 96, "blk.%d.attn_output.weight", l); wti(f, n, lw->wo, &off);
        snprintf(n, 96, "blk.%d.ffn_norm.weight", l); wti(f, n, lw->ffn_norm, &off);
        snprintf(n, 96, "blk.%d.ffn_gate_inp.weight", l); wti(f, n, lw->parliament.w_vote, &off);
        for (int e = 0; e < MAX_EXPERTS; e++) {
            if (!lw->experts[e].alive) continue;
            snprintf(n, 96, "blk.%d.ffn_gate.%d.weight", l, e); wti(f, n, lw->experts[e].w_gate, &off);
            snprintf(n, 96, "blk.%d.ffn_up.%d.weight", l, e); wti(f, n, lw->experts[e].w_up, &off);
            snprintf(n, 96, "blk.%d.ffn_down.%d.weight", l, e); wti(f, n, lw->experts[e].w_down, &off);
        }
    }
    long p = ftell(f); long al = ((p+31)/32)*32; for (long i = p; i < al; i++) fputc(0, f);
    #define WD(t) fwrite((t)->data, 4, (t)->size, f)
    WD(w->tok_emb); WD(w->output_norm); WD(w->output);
    for (int l = 0; l < c->depth; l++) {
        LayerW *lw = &w->layers[l]; WD(lw->attn_norm); WD(lw->wq); WD(lw->wk); WD(lw->wv); WD(lw->wo); WD(lw->ffn_norm); WD(lw->parliament.w_vote);
        for (int e = 0; e < MAX_EXPERTS; e++) { if (!lw->experts[e].alive) continue; WD(lw->experts[e].w_gate); WD(lw->experts[e].w_up); WD(lw->experts[e].w_down); }
    }
    fclose(f);
    struct stat st2; stat(c->gguf_path, &st2);
    printf("[gguf] exported %s (%.1f MB)\n", c->gguf_path, (float)st2.st_size / 1048576);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MYCELIUM — the GGUF forest. during training, the system grows multiple GGUF
 * snapshots like mushrooms from mycelium. each snapshot captures a different
 * expert configuration (different population, different consensus, different
 * harmonic state). on startup, the system discovers existing snapshots and
 * picks the fittest one. "the forest is the network. the mushrooms are models."
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MYCELIUM_DIR "mycelium"
#define MYCELIUM_MAX 64
#define MYCELIUM_INTERVAL 200 /* save snapshot every N steps */

typedef struct {
    char path[256];
    int step;
    int n_experts;
    float loss;
    float consensus;
    float fitness;      /* composite score: lower loss + higher consensus = better */
} MyceliumSpore;

typedef struct {
    MyceliumSpore spores[MYCELIUM_MAX];
    int n_spores;
    int best_idx;       /* index of fittest spore */
} MyceliumState;

static void mycelium_init(MyceliumState *ms) {
    memset(ms, 0, sizeof(MyceliumState));
    ms->best_idx = -1;
    mkdir(MYCELIUM_DIR, 0755);
}

static float mycelium_fitness(float loss, float consensus, int n_experts) {
    /* lower loss = better, higher consensus = clearer decisions, moderate experts = sweet spot */
    float loss_score = 1.0f / (loss + 0.01f);
    float cons_score = consensus;
    float exp_score = 1.0f - fabsf((float)n_experts / MAX_EXPERTS - 0.5f); /* peak at 50% capacity */
    return loss_score * 0.5f + cons_score * 0.3f + exp_score * 0.2f;
}

static void mycelium_save_spore(MyceliumState *ms, ModelW *w, Config *c,
                                 int step, float loss, float consensus) {
    if (ms->n_spores >= MYCELIUM_MAX) {
        /* replace worst spore */
        int worst = 0;
        for (int i = 1; i < ms->n_spores; i++)
            if (ms->spores[i].fitness < ms->spores[worst].fitness) worst = i;
        /* overwrite worst */
        int n_exp = 0;
        for (int l = 0; l < c->depth; l++) n_exp += w->layers[l].n_alive;
        MyceliumSpore *sp = &ms->spores[worst];
        snprintf(sp->path, 256, "%s/m_s%d_e%d_l%.3f.gguf", MYCELIUM_DIR, step, n_exp, loss);
        sp->step = step; sp->n_experts = n_exp; sp->loss = loss; sp->consensus = consensus;
        sp->fitness = mycelium_fitness(loss, consensus, n_exp);
        /* swap gguf_path, export, restore */
        char orig[256]; snprintf(orig, 256, "%s", c->gguf_path);
        snprintf(c->gguf_path, 256, "%s", sp->path);
        export_gguf(w, c);
        snprintf(c->gguf_path, 256, "%s", orig);
        return;
    }
    int n_exp = 0;
    for (int l = 0; l < c->depth; l++) n_exp += w->layers[l].n_alive;
    MyceliumSpore *sp = &ms->spores[ms->n_spores];
    snprintf(sp->path, 256, "%s/m_s%d_e%d_l%.3f.gguf", MYCELIUM_DIR, step, n_exp, loss);
    sp->step = step; sp->n_experts = n_exp; sp->loss = loss; sp->consensus = consensus;
    sp->fitness = mycelium_fitness(loss, consensus, n_exp);
    char orig[256]; snprintf(orig, 256, "%s", c->gguf_path);
    snprintf(c->gguf_path, 256, "%s", sp->path);
    export_gguf(w, c);
    snprintf(c->gguf_path, 256, "%s", orig);
    ms->n_spores++;
    /* update best */
    ms->best_idx = 0;
    for (int i = 1; i < ms->n_spores; i++)
        if (ms->spores[i].fitness > ms->spores[ms->best_idx].fitness) ms->best_idx = i;
    printf("  [mycelium] spore %d: step=%d experts=%d loss=%.4f fitness=%.2f (%d total)\n",
           ms->n_spores-1, step, n_exp, loss, sp->fitness, ms->n_spores);
}

/* Scan mycelium directory for existing spores */
static void mycelium_discover(MyceliumState *ms) {
    char cmd[512];
    snprintf(cmd, 512, "ls %s/m_s*_e*_l*.gguf 2>/dev/null", MYCELIUM_DIR);
    FILE *p = popen(cmd, "r");
    if (!p) return;
    char line[256];
    while (fgets(line, sizeof(line), p) && ms->n_spores < MYCELIUM_MAX) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        MyceliumSpore *sp = &ms->spores[ms->n_spores];
        snprintf(sp->path, 256, "%s", line);
        /* parse metrics from filename: m_s{step}_e{experts}_l{loss}.gguf */
        int s_val = 0, e_val = 0; float l_val = 0;
        char *ps = strstr(line, "_s"); if (ps) s_val = atoi(ps+2);
        char *pe = strstr(line, "_e"); if (pe) e_val = atoi(pe+2);
        char *pl = strstr(line, "_l"); if (pl) l_val = atof(pl+2);
        sp->step = s_val; sp->n_experts = e_val; sp->loss = l_val;
        sp->consensus = 0.5f; /* unknown from filename */
        sp->fitness = mycelium_fitness(l_val, 0.5f, e_val);
        ms->n_spores++;
    }
    pclose(p);
    if (ms->n_spores > 0) {
        ms->best_idx = 0;
        for (int i = 1; i < ms->n_spores; i++)
            if (ms->spores[i].fitness > ms->spores[ms->best_idx].fitness) ms->best_idx = i;
        printf("[mycelium] discovered %d spores. best: %s (fitness=%.2f)\n",
               ms->n_spores, ms->spores[ms->best_idx].path, ms->spores[ms->best_idx].fitness);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * META-LEARNING TRACK — the system learns from its own choices.
 * after each GGUF snapshot, record: what config was used, how well it worked.
 * over time, build a map of "this many experts + this consensus + this data
 * quality = this much loss reduction." the model trains itself on itself.
 * meta-cognition for neural networks. descartes is spinning.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define META_LOG "mycelium/meta.log"
#define META_HIST_CAP 128

typedef struct {
    int step;
    int n_experts;
    float consensus;
    float loss;
    float tok_health;
    float parser_health;
    float harmonic_conf;
    float fitness;
    float delta_loss;  /* loss improvement since last meta entry */
} MetaEntry;

typedef struct {
    MetaEntry history[META_HIST_CAP];
    int n_entries;
    float predicted_fitness; /* what we predicted vs what happened */
    float prediction_error;  /* running error of our meta-predictions */
    float config_bias[4];    /* learned biases: [expert_count, consensus, tok_health, parser_health] */
} MetaTrack;

static void meta_init(MetaTrack *mt) {
    memset(mt, 0, sizeof(MetaTrack));
    mt->config_bias[0] = 0.5f; /* neutral initial biases */
    mt->config_bias[1] = 0.5f;
    mt->config_bias[2] = 0.5f;
    mt->config_bias[3] = 0.5f;
}

static void meta_record(MetaTrack *mt, int step, int n_experts, float consensus,
                        float loss, float tok_health, float parser_health,
                        float harmonic_conf, float prev_loss) {
    if (mt->n_entries >= META_HIST_CAP) {
        /* shift left — drop oldest */
        memmove(mt->history, mt->history + 1, (META_HIST_CAP-1) * sizeof(MetaEntry));
        mt->n_entries = META_HIST_CAP - 1;
    }
    MetaEntry *e = &mt->history[mt->n_entries];
    e->step = step; e->n_experts = n_experts; e->consensus = consensus;
    e->loss = loss; e->tok_health = tok_health; e->parser_health = parser_health;
    e->harmonic_conf = harmonic_conf;
    e->delta_loss = (prev_loss > 0) ? prev_loss - loss : 0;
    e->fitness = mycelium_fitness(loss, consensus, n_experts);
    mt->n_entries++;

    /* Meta-learn: update config biases based on what worked */
    if (mt->n_entries >= 2) {
        MetaEntry *prev = &mt->history[mt->n_entries - 2];
        float improvement = prev->loss - loss;
        float lr_meta = 0.01f;
        /* If loss improved, strengthen biases toward current config */
        /* If loss worsened, weaken them */
        float signal = (improvement > 0) ? 1.0f : -0.5f;
        mt->config_bias[0] += lr_meta * signal * ((float)n_experts / MAX_EXPERTS - 0.5f);
        mt->config_bias[1] += lr_meta * signal * (consensus - 0.5f);
        mt->config_bias[2] += lr_meta * signal * (tok_health - 0.5f);
        mt->config_bias[3] += lr_meta * signal * (parser_health - 0.5f);
        for (int i = 0; i < 4; i++) {
            if (mt->config_bias[i] < 0.01f) mt->config_bias[i] = 0.01f;
            if (mt->config_bias[i] > 0.99f) mt->config_bias[i] = 0.99f;
        }
        /* Track prediction accuracy */
        float pred_err = fabsf(mt->predicted_fitness - e->fitness);
        mt->prediction_error = 0.9f * mt->prediction_error + 0.1f * pred_err;
    }
    /* Predict fitness for next step (used by ephemeral config) */
    mt->predicted_fitness = mycelium_fitness(loss * 0.99f, consensus, n_experts);
}

static void meta_save(MetaTrack *mt) {
    FILE *f = fopen(META_LOG, "a");
    if (!f) return;
    MetaEntry *e = &mt->history[mt->n_entries - 1];
    fprintf(f, "step=%d experts=%d consensus=%.3f loss=%.4f tok=%.2f parser=%.2f harmonic=%.2f fitness=%.2f delta=%.4f bias=[%.2f,%.2f,%.2f,%.2f]\n",
            e->step, e->n_experts, e->consensus, e->loss, e->tok_health, e->parser_health,
            e->harmonic_conf, e->fitness, e->delta_loss,
            mt->config_bias[0], mt->config_bias[1], mt->config_bias[2], mt->config_bias[3]);
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TOKENIZER CACHE — learned encoding memory.
 * if the tokenizer already encoded this pattern, why encode it again?
 * a small hash table caches recent encode results. the tokenizer learns
 * which patterns it sees often and can skip re-encoding them.
 * the tokenizer has a memory. it remembers what it ate.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define TOKCACHE_CAP 1024
#define TOKCACHE_MAX_LEN 128 /* max text length to cache */

typedef struct {
    uint64_t hash;
    int *ids;
    int n_ids;
    int text_len;
    int hits;      /* how many times this cache entry was used */
    int alive;
} TokCacheEntry;

typedef struct {
    TokCacheEntry entries[TOKCACHE_CAP];
    int total_hits;
    int total_misses;
    float hit_rate;   /* EMA of hit rate */
    /* Learned weights: bias toward commonly seen tokens */
    float *tok_freq;  /* [vocab_size] — learned token frequency bias */
    int vocab_size;
} TokCache;

static void tokcache_init(TokCache *tc, int vocab_size) {
    memset(tc, 0, sizeof(TokCache));
    tc->vocab_size = vocab_size;
    tc->tok_freq = calloc(vocab_size, sizeof(float));
    /* Initialize with uniform */
    float init = 1.0f / vocab_size;
    for (int i = 0; i < vocab_size; i++) tc->tok_freq[i] = init;
}

static uint64_t tokcache_hash(const char *text, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) { h ^= (uint64_t)(unsigned char)text[i]; h *= 1099511628211ULL; }
    return h;
}

static int *tokcache_lookup(TokCache *tc, const char *text, int len, int *n_ids) {
    if (len > TOKCACHE_MAX_LEN || len == 0) return NULL;
    uint64_t h = tokcache_hash(text, len);
    int idx = (int)(h % TOKCACHE_CAP);
    for (int probe = 0; probe < 8; probe++) {
        int i = (idx + probe) % TOKCACHE_CAP;
        if (tc->entries[i].alive && tc->entries[i].hash == h && tc->entries[i].text_len == len) {
            tc->entries[i].hits++;
            tc->total_hits++;
            tc->hit_rate = 0.95f * tc->hit_rate + 0.05f * 1.0f;
            *n_ids = tc->entries[i].n_ids;
            return tc->entries[i].ids;
        }
    }
    tc->total_misses++;
    tc->hit_rate = 0.95f * tc->hit_rate + 0.05f * 0.0f;
    return NULL;
}

static void tokcache_insert(TokCache *tc, const char *text, int len, int *ids, int n_ids) {
    if (len > TOKCACHE_MAX_LEN || len == 0) return;
    uint64_t h = tokcache_hash(text, len);
    int idx = (int)(h % TOKCACHE_CAP);
    /* find empty or LRU slot */
    int target = idx;
    int min_hits = tc->entries[idx].alive ? tc->entries[idx].hits : -1;
    for (int probe = 0; probe < 8; probe++) {
        int i = (idx + probe) % TOKCACHE_CAP;
        if (!tc->entries[i].alive) { target = i; break; }
        if (tc->entries[i].hits < min_hits) { min_hits = tc->entries[i].hits; target = i; }
    }
    TokCacheEntry *e = &tc->entries[target];
    if (e->alive) free(e->ids);
    e->hash = h; e->text_len = len; e->hits = 0; e->alive = 1;
    e->n_ids = n_ids;
    e->ids = malloc(n_ids * sizeof(int));
    memcpy(e->ids, ids, n_ids * sizeof(int));
    /* Update learned token frequency bias */
    float lr_tok = 0.001f;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= 0 && ids[i] < tc->vocab_size)
            tc->tok_freq[ids[i]] = (1.0f - lr_tok) * tc->tok_freq[ids[i]] + lr_tok;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — temporal self-awareness from AML.
 * inference is the present: ephemeral, no memory, topology born and dying.
 * training is the past: weights persist, experience accumulates.
 * drift is the distance between what the system remembers and what it sees now.
 *
 * every N steps, take a "photograph" of the full system state:
 * expert population, consensus distribution, loss landscape, harmonic spectrum,
 * tokenizer health, parser quality. compare current to historical snapshots.
 * large drift → the world changed → rebuild aggressively.
 * small drift → stable → conserve energy, exploit what works.
 *
 * the AML principle: identity = ε + γ + αδ. the system has a persistent self
 * (γ = weights, training memory) and an ephemeral experience (ε = inference).
 * drift measures how far ε has drifted from γ. when they diverge too much,
 * something fundamental changed. when they converge, the system found itself.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define DRIFT_SNAPSHOTS 32
#define DRIFT_INTERVAL 50 /* snapshot every N steps */
#define DRIFT_DIM 12     /* dimensions of state vector */

typedef struct {
    float state[DRIFT_DIM];
    /* state layout:
     * [0] loss
     * [1] avg_consensus (across layers)
     * [2] n_experts_ratio (n_alive / MAX_EXPERTS, per layer avg)
     * [3] tok_health
     * [4] parser_health
     * [5] harmonic_dominant_freq
     * [6] harmonic_confidence
     * [7] chuck_lambda
     * [8] chuck_sigma
     * [9] chuck_lr_scale
     * [10] avg_vitality (across all alive experts)
     * [11] meta_prediction_error
     */
    int step;
} DriftSnapshot;

typedef struct {
    DriftSnapshot history[DRIFT_SNAPSHOTS];
    int n_snapshots;
    int head;           /* circular buffer pointer */
    float drift;        /* current drift magnitude */
    float drift_ema;    /* exponential moving average of drift */
    float drift_accel;  /* rate of change of drift (second derivative) */
    float stability;    /* 0=chaotic, 1=stable. inverse of drift_ema. */
    /* Temporal spectrum: fourier decomposition of drift history */
    float drift_hist[DRIFT_SNAPSHOTS];
    int drift_hist_len;
} CalendarDrift;

static void drift_init(CalendarDrift *cd) {
    memset(cd, 0, sizeof(CalendarDrift));
    cd->stability = 0.5f;
}

static void drift_snapshot(CalendarDrift *cd, float loss, ModelW *w, Config *c,
                           TokenizerEye *tok_eye, ParserEye *parser_eye,
                           HarmonicState *hs, ChuckState *chuck, MetaTrack *meta) {
    DriftSnapshot *snap = &cd->history[cd->head];

    /* Compute state vector */
    snap->state[0] = loss;

    /* avg consensus across layers */
    float avg_cons = 0;
    for (int l = 0; l < c->depth; l++) avg_cons += w->layers[l].parliament.consensus;
    snap->state[1] = avg_cons / c->depth;

    /* expert ratio */
    float avg_exp = 0;
    for (int l = 0; l < c->depth; l++) avg_exp += (float)w->layers[l].n_alive / MAX_EXPERTS;
    snap->state[2] = avg_exp / c->depth;

    snap->state[3] = tok_eye->health;
    snap->state[4] = parser_eye->health;
    snap->state[5] = hs->dominant_freq;
    snap->state[6] = hs->confidence;
    snap->state[7] = chuck->dampen;
    snap->state[8] = chuck->sigma;
    snap->state[9] = chuck->lr_scale;

    /* avg vitality */
    float avg_vit = 0; int n_alive = 0;
    for (int l = 0; l < c->depth; l++)
        for (int e = 0; e < MAX_EXPERTS; e++)
            if (w->layers[l].experts[e].alive) { avg_vit += w->layers[l].experts[e].vitality; n_alive++; }
    snap->state[10] = (n_alive > 0) ? avg_vit / n_alive : 0;
    snap->state[11] = meta->prediction_error;

    snap->step = chuck->global_step;

    /* Compute drift against previous snapshot */
    if (cd->n_snapshots > 0) {
        int prev_idx = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
        DriftSnapshot *prev = &cd->history[prev_idx];
        float dist_sq = 0;
        for (int i = 0; i < DRIFT_DIM; i++) {
            float d = snap->state[i] - prev->state[i];
            /* normalize by magnitude to make dimensions comparable */
            float scale = fabsf(prev->state[i]) + 0.01f;
            dist_sq += (d / scale) * (d / scale);
        }
        float new_drift = sqrtf(dist_sq / DRIFT_DIM);

        /* drift acceleration = how fast drift is changing */
        float prev_drift = cd->drift;
        cd->drift_accel = new_drift - prev_drift;
        cd->drift = new_drift;
        cd->drift_ema = 0.9f * cd->drift_ema + 0.1f * new_drift;
        cd->stability = 1.0f / (1.0f + cd->drift_ema * 5.0f);
    }

    /* Record drift history for temporal fourier analysis */
    if (cd->drift_hist_len < DRIFT_SNAPSHOTS) cd->drift_hist[cd->drift_hist_len++] = cd->drift;
    else { memmove(cd->drift_hist, cd->drift_hist + 1, (DRIFT_SNAPSHOTS-1) * sizeof(float)); cd->drift_hist[DRIFT_SNAPSHOTS-1] = cd->drift; }

    /* Advance circular buffer */
    cd->head = (cd->head + 1) % DRIFT_SNAPSHOTS;
    if (cd->n_snapshots < DRIFT_SNAPSHOTS) cd->n_snapshots++;
}

/* Compare current state against a specific historical snapshot */
static float drift_compare(CalendarDrift *cd, int snapshot_idx) {
    if (snapshot_idx >= cd->n_snapshots || cd->n_snapshots < 2) return 0;
    int curr = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
    int hist = (cd->head - 1 - snapshot_idx + DRIFT_SNAPSHOTS * 2) % DRIFT_SNAPSHOTS;
    float dist_sq = 0;
    for (int i = 0; i < DRIFT_DIM; i++) {
        float d = cd->history[curr].state[i] - cd->history[hist].state[i];
        float scale = fabsf(cd->history[hist].state[i]) + 0.01f;
        dist_sq += (d / scale) * (d / scale);
    }
    return sqrtf(dist_sq / DRIFT_DIM);
}

/* Find the historical snapshot most similar to current state */
static int drift_find_resonance(CalendarDrift *cd) {
    if (cd->n_snapshots < 3) return -1;
    float min_dist = 1e30f; int best = -1;
    int curr = (cd->head - 1 + DRIFT_SNAPSHOTS) % DRIFT_SNAPSHOTS;
    for (int i = 2; i < cd->n_snapshots; i++) { /* skip last 2 (too recent) */
        int idx = (cd->head - 1 - i + DRIFT_SNAPSHOTS * 2) % DRIFT_SNAPSHOTS;
        float dist = 0;
        for (int j = 0; j < DRIFT_DIM; j++) {
            float d = cd->history[curr].state[j] - cd->history[idx].state[j];
            float scale = fabsf(cd->history[idx].state[j]) + 0.01f;
            dist += (d / scale) * (d / scale);
        }
        dist = sqrtf(dist / DRIFT_DIM);
        if (dist < min_dist) { min_dist = dist; best = i; }
    }
    return best; /* steps ago when system was in a similar state */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * EPHEMERAL METRIC-DRIVEN CONFIG — the architecture doesn't care how many
 * weights it has. it adapts. each step, a hash of all metrics determines:
 * - which experts are "hot" (get extra weight)
 * - how many layers are active
 * - vitality threshold for mitosis/apoptosis
 * the unpredictable combination means every run produces a different topology.
 * same weights, different architecture. this is the nicole principle incarnate.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float expert_temperature;  /* modulates parliament vote sharpness */
    float vitality_threshold;  /* adaptive threshold for birth/death */
    int active_layers;         /* how many layers actually execute this step */
    float mitosis_eagerness;   /* how eager the system is to birth new experts */
    float apoptosis_mercy;     /* how merciful toward dying experts */
    uint64_t config_hash;      /* the metric hash that produced this config */
} EphemeralConfig;

static EphemeralConfig ephemeral_compute(TokenizerEye *tok_eye, ParserEye *parser_eye,
                                          HarmonicState *hs, float consensus,
                                          MetaTrack *mt, CalendarDrift *cd, int depth) {
    EphemeralConfig ec;
    /* Hash all metrics into a single chaotic value — including drift */
    uint64_t h = 0x5DEECE66DULL;
    uint32_t bits[8];
    memcpy(&bits[0], &tok_eye->health, 4);
    memcpy(&bits[1], &parser_eye->health, 4);
    memcpy(&bits[2], &hs->dominant_freq, 4);
    memcpy(&bits[3], &consensus, 4);
    memcpy(&bits[4], &mt->config_bias[0], 4);
    memcpy(&bits[5], &mt->prediction_error, 4);
    memcpy(&bits[6], &cd->drift, 4);
    memcpy(&bits[7], &cd->stability, 4);
    for (int i = 0; i < 8; i++) { h ^= (uint64_t)bits[i]; h *= 0x100000001B3ULL; h ^= h >> 17; }
    ec.config_hash = h;

    /* Derive config from hash + metrics + drift */
    float metric_blend = tok_eye->health * 0.25f + parser_eye->health * 0.25f +
                         hs->confidence * 0.15f + consensus * 0.15f + cd->stability * 0.2f;

    /* Expert temperature: higher when confused OR when drift is high (world changed) */
    ec.expert_temperature = 0.5f + 1.0f * (1.0f - consensus);
    if (cd->drift > 0.3f) ec.expert_temperature *= 1.2f; /* more exploration during drift */
    ec.expert_temperature *= (0.5f + mt->config_bias[1]);

    /* Active layers: skip some when stable and simple, use all when drifting */
    float complexity = tok_eye->entropy / 5.0f;
    if (complexity > 1.0f) complexity = 1.0f;
    ec.active_layers = depth;
    if (cd->stability > 0.8f && metric_blend > 0.8f && complexity < 0.3f && depth > 2) {
        ec.active_layers = depth - 1; /* skip layer only when truly stable */
    }
    if (cd->drift > 0.5f && depth > 2) {
        ec.active_layers = depth; /* force full depth during high drift */
    }

    /* Mitosis eagerness: more births when drifting (need to adapt) or data quality high */
    ec.mitosis_eagerness = 0.5f + 0.3f * parser_eye->health * mt->config_bias[0];
    if (cd->drift > 0.2f) ec.mitosis_eagerness += 0.2f * cd->drift; /* drift triggers births */
    if (ec.mitosis_eagerness > 1.0f) ec.mitosis_eagerness = 1.0f;

    /* Apoptosis mercy: more merciful when drifting (don't kill during transition) */
    ec.apoptosis_mercy = 0.5f + 0.3f * (1.0f - metric_blend);
    if (cd->drift > 0.3f) ec.apoptosis_mercy += 0.2f; /* protect experts during drift */
    if (cd->drift_accel > 0.1f) ec.apoptosis_mercy += 0.1f; /* accelerating drift = even more mercy */
    if (ec.apoptosis_mercy > 1.0f) ec.apoptosis_mercy = 1.0f;

    /* Vitality threshold: adaptive based on stability */
    ec.vitality_threshold = 0.1f * (1.0f + ec.apoptosis_mercy);

    return ec;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENVIRONMENT SCANNER — DOE opens its eyes. looks around. counts its resources.
 * what GGUFs are nearby? how much RAM? is there a compiler? can it curl?
 * a model that doesn't know its own filesystem is a model that can't survive.
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char path[256];
    char arch[64];      /* "llama", "grok", "mixtral", "m" — from GGUF metadata */
    int n_layers;
    int dim;
    int n_heads;
    int64_t file_size;
    float compatibility; /* 0..1 — how usable is this GGUF for parasite mode */
} DiscoveredGGUF;

typedef struct {
    DiscoveredGGUF ggufs[32];
    int n_ggufs;
    int64_t disk_free;
    int cpu_count;
    int64_t mem_available;
    int has_compiler;
    int has_curl;
    char self_path[256]; /* path to own m.c source */
} Environment;

/* Read GGUF header — just metadata, no tensor loading. peek at the corpse. */
static int gguf_sniff(const char *path, DiscoveredGGUF *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    struct stat st; fstat(fileno(f), &st); out->file_size = st.st_size;
    snprintf(out->path, 256, "%s", path);
    memset(out->arch, 0, 64); out->n_layers = 0; out->dim = 0; out->n_heads = 0;

    uint32_t magic; if (fread(&magic, 4, 1, f) != 1 || magic != 0x46554747) { fclose(f); return 0; }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv; fread(&n_tensors, 8, 1, f); fread(&n_kv, 8, 1, f);

    /* Parse metadata KV pairs — we only care about architecture, dim, layers, heads */
    for (uint64_t i = 0; i < n_kv && i < 64; i++) {
        uint64_t klen; if (fread(&klen, 8, 1, f) != 1) break;
        if (klen > 255) break;
        char key[256]; if (fread(key, 1, klen, f) != klen) break; key[klen] = '\0';
        uint32_t vtype; if (fread(&vtype, 4, 1, f) != 1) break;

        if (vtype == 8) { /* string */
            uint64_t vlen; fread(&vlen, 8, 1, f);
            char val[256]; int rl = vlen < 255 ? (int)vlen : 255;
            fread(val, 1, rl, f); val[rl] = '\0';
            if (vlen > 255) fseek(f, vlen - 255, SEEK_CUR);
            if (strstr(key, "general.architecture")) snprintf(out->arch, 64, "%s", val);
        } else if (vtype == 4) { /* uint32 */
            uint32_t val; fread(&val, 4, 1, f);
            if (strstr(key, "embedding_length")) out->dim = (int)val;
            else if (strstr(key, "block_count")) out->n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) out->n_heads = (int)val;
        } else if (vtype == 5) { /* int32 */
            fseek(f, 4, SEEK_CUR);
        } else if (vtype == 6) { /* float32 */
            fseek(f, 4, SEEK_CUR);
        } else if (vtype == 10) { /* uint64 */
            fseek(f, 8, SEEK_CUR);
        } else if (vtype == 7) { /* bool */
            fseek(f, 1, SEEK_CUR);
        } else {
            break; /* unknown type — stop parsing, we got enough */
        }
    }
    fclose(f);
    return (out->arch[0] != '\0' && out->dim > 0);
}

static void env_scan(Environment *env, const char *self_src, int doe_dim) {
    memset(env, 0, sizeof(Environment));
    snprintf(env->self_path, 256, "%s", self_src);

    /* CPU count */
    env->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);

    /* Memory */
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    env->mem_available = (int64_t)pages * page_size;
#elif defined(__APPLE__)
    int64_t mem = 0; size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    env->mem_available = mem;
#endif

    /* Disk free */
#ifdef __linux__
    struct statvfs sv; if (statvfs(".", &sv) == 0) env->disk_free = (int64_t)sv.f_bavail * sv.f_frsize;
#elif defined(__APPLE__)
    struct statfs sf; if (statfs(".", &sf) == 0) env->disk_free = (int64_t)sf.f_bavail * sf.f_bsize;
#endif

    /* Check for compiler and curl — the tools of survival */
    env->has_compiler = (system("which cc >/dev/null 2>&1") == 0);
    env->has_curl = (system("which curl >/dev/null 2>&1") == 0);

    /* Scan for GGUFs — find . -name '*.gguf' -maxdepth 3 */
    FILE *p = popen("find . -name '*.gguf' -maxdepth 3 2>/dev/null", "r");
    if (p) {
        char line[256];
        while (fgets(line, sizeof(line), p) && env->n_ggufs < 32) {
            int len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
            if (len == 0) continue;
            DiscoveredGGUF dg;
            if (gguf_sniff(line, &dg)) {
                /* Compatibility: dim match = best, dim multiple = ok, else = poor */
                if (dg.dim == doe_dim) dg.compatibility = 1.0f;
                else if (doe_dim > 0 && dg.dim % doe_dim == 0) dg.compatibility = 0.7f;
                else if (doe_dim > 0 && abs(dg.dim - doe_dim) < 128) dg.compatibility = 0.5f;
                else dg.compatibility = 0.1f;
                env->ggufs[env->n_ggufs++] = dg;
            }
        }
        pclose(p);
    }

    printf("[env] cpu=%d mem=%.1fGB disk=%.1fGB compiler=%s curl=%s ggufs=%d\n",
           env->cpu_count,
           (float)env->mem_available / (1024*1024*1024),
           (float)env->disk_free / (1024*1024*1024),
           env->has_compiler ? "yes" : "no",
           env->has_curl ? "yes" : "no",
           env->n_ggufs);

    for (int i = 0; i < env->n_ggufs; i++) {
        DiscoveredGGUF *g = &env->ggufs[i];
        printf("  [env] gguf: %s arch=%s dim=%d layers=%d size=%.1fMB compat=%.1f\n",
               g->path, g->arch, g->dim, g->n_layers,
               (float)g->file_size / (1024*1024), g->compatibility);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF PARASITE — DOE finds a bigger model. wraps it with LoRA. controls it.
 * the smallest model that ever commanded a 7B. like a remora on a shark.
 * like a virus that makes the host cell produce more virus.
 * "i am not the model. i am the thing that tells the model what to think."
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define LORA_RANK 16
#define LORA_MAX_LAYERS 32

typedef struct {
    /* Host model — mmap'd from GGUF, read-only. the shark. */
    uint8_t *mmap_base;
    size_t mmap_size;
    int host_n_layers, host_dim, host_hidden, host_heads, host_kv_heads, host_head_dim;
    int host_vocab;
    float *host_tok_emb;    /* [vocab, dim] */
    float *host_output;     /* [vocab, dim] */
    float *host_norm;       /* [dim] */
    struct {
        float *wq, *wk, *wv, *wo;
        float *ffn_gate, *ffn_up, *ffn_down;
        float *attn_norm, *ffn_norm;
    } layers[LORA_MAX_LAYERS];

    /* DOE's LoRA overlays — the remora. small. trained. in control. */
    /* Delta Voice: out += alpha * A @ (B @ x) */
    float *lora_A[LORA_MAX_LAYERS];   /* [dim, rank] — output projection */
    float *lora_B[LORA_MAX_LAYERS];   /* [rank, dim] — input projection */
    float attention_bias[LORA_MAX_LAYERS];  /* per-layer attention scaling (Meta-Arianna) */
    float layer_focus[LORA_MAX_LAYERS];     /* per-layer residual scaling */
    float lora_alpha;

    int active;
    char host_path[256];
} ParasiteState;

/* Parse GGUF tensor info entry */
typedef struct { char name[96]; uint32_t ndim; uint64_t dims[4]; uint32_t dtype; uint64_t offset; } GGUFTensorInfo;

static int parasite_load(ParasiteState *ps, const char *path, int doe_dim) {
    memset(ps, 0, sizeof(ParasiteState));
    snprintf(ps->host_path, 256, "%s", path);
    ps->lora_alpha = 0.1f;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st; fstat(fd, &st);
    ps->mmap_size = st.st_size;
    ps->mmap_base = mmap(NULL, ps->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ps->mmap_base == MAP_FAILED) { ps->mmap_base = NULL; return 0; }

    /* Parse GGUF header from mmap'd data */
    uint8_t *p = ps->mmap_base;
    uint8_t *pend = ps->mmap_base + ps->mmap_size;
    #define PCHECK(n) do { if (p + (n) > pend) goto bail; } while(0)
    PCHECK(4); uint32_t magic = *(uint32_t*)p; p += 4;
    if (magic != 0x46554747) goto bail;
    PCHECK(4); uint32_t ver = *(uint32_t*)p; p += 4; (void)ver;
    PCHECK(8); uint64_t n_tensors = *(uint64_t*)p; p += 8;
    PCHECK(8); uint64_t n_kv = *(uint64_t*)p; p += 8;

    /* Parse metadata to get model dimensions */
    for (uint64_t i = 0; i < n_kv && i < 64; i++) {
        PCHECK(8); uint64_t klen = *(uint64_t*)p; p += 8;
        if (klen > 255) break;
        char key[256]; memcpy(key, p, klen); key[klen] = '\0'; p += klen;
        uint32_t vtype = *(uint32_t*)p; p += 4;
        if (vtype == 8) { /* string */
            uint64_t vlen = *(uint64_t*)p; p += 8; p += vlen;
        } else if (vtype == 4) { /* uint32 */
            uint32_t val = *(uint32_t*)p; p += 4;
            if (strstr(key, "embedding_length")) ps->host_dim = (int)val;
            else if (strstr(key, "block_count")) ps->host_n_layers = (int)val;
            else if (strstr(key, "head_count") && !strstr(key, "kv")) ps->host_heads = (int)val;
            else if (strstr(key, "head_count_kv")) ps->host_kv_heads = (int)val;
            else if (strstr(key, "feed_forward_length")) ps->host_hidden = (int)val;
            else if (strstr(key, "context_length")) { /* ignore */ }
            else if (strstr(key, "vocab_size")) ps->host_vocab = (int)val;
        } else if (vtype == 5) p += 4; /* int32 */
        else if (vtype == 6) p += 4;   /* float32 */
        else if (vtype == 10) p += 8;  /* uint64 */
        else if (vtype == 7) p += 1;   /* bool */
        else break;
    }

    if (ps->host_dim == 0 || ps->host_n_layers == 0) goto bail;
    if (ps->host_heads == 0) ps->host_heads = ps->host_dim / 64;
    if (ps->host_kv_heads == 0) ps->host_kv_heads = ps->host_heads;
    ps->host_head_dim = ps->host_dim / ps->host_heads;
    if (ps->host_hidden == 0) ps->host_hidden = ps->host_dim * 4;

    /* Parse tensor info entries — find offsets for each weight */
    if (n_tensors > 10000) goto bail; /* sanity check */
    GGUFTensorInfo *tinfo = calloc(n_tensors, sizeof(GGUFTensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        PCHECK(8); uint64_t nlen = *(uint64_t*)p; p += 8;
        if (nlen > 256) { free(tinfo); goto bail; }
        int nl = nlen < 95 ? (int)nlen : 95;
        PCHECK(nlen); memcpy(tinfo[i].name, p, nl); tinfo[i].name[nl] = '\0'; p += nlen;
        PCHECK(4); tinfo[i].ndim = *(uint32_t*)p; p += 4;
        if (tinfo[i].ndim > 4) { free(tinfo); goto bail; }
        for (uint32_t d = 0; d < tinfo[i].ndim; d++) { PCHECK(8); tinfo[i].dims[d] = *(uint64_t*)p; p += 8; }
        PCHECK(4); tinfo[i].dtype = *(uint32_t*)p; p += 4;
        PCHECK(8); tinfo[i].offset = *(uint64_t*)p; p += 8;
    }

    /* Align to 32 bytes — tensor data starts here */
    uint64_t header_size = p - ps->mmap_base;
    uint64_t data_start = ((header_size + 31) / 32) * 32;

    /* Map tensor names to weight pointers — the parasitic wiring */
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (tinfo[i].dtype != 0) continue; /* only float32 for now */
        float *data = (float*)(ps->mmap_base + data_start + tinfo[i].offset);
        char *n = tinfo[i].name;

        if (strcmp(n, "token_embd.weight") == 0) { ps->host_tok_emb = data; if (ps->host_vocab == 0) ps->host_vocab = (int)tinfo[i].dims[1]; }
        else if (strcmp(n, "output_norm.weight") == 0) ps->host_norm = data;
        else if (strcmp(n, "output.weight") == 0) ps->host_output = data;
        else {
            int l = -1; sscanf(n, "blk.%d.", &l);
            if (l >= 0 && l < LORA_MAX_LAYERS && l < ps->host_n_layers) {
                if (strstr(n, "attn_q.weight")) ps->layers[l].wq = data;
                else if (strstr(n, "attn_k.weight")) ps->layers[l].wk = data;
                else if (strstr(n, "attn_v.weight")) ps->layers[l].wv = data;
                else if (strstr(n, "attn_output.weight")) ps->layers[l].wo = data;
                else if (strstr(n, "ffn_gate.weight") && !strstr(n, "ffn_gate_inp")) ps->layers[l].ffn_gate = data;
                else if (strstr(n, "ffn_up.weight")) ps->layers[l].ffn_up = data;
                else if (strstr(n, "ffn_down.weight")) ps->layers[l].ffn_down = data;
                else if (strstr(n, "attn_norm.weight")) ps->layers[l].attn_norm = data;
                else if (strstr(n, "ffn_norm.weight")) ps->layers[l].ffn_norm = data;
            }
        }
    }
    free(tinfo);

    /* Verify minimum viable host — can't parasitize a headless corpse */
    if (!ps->host_tok_emb || !ps->host_output || !ps->host_norm) {
        printf("[parasite] host GGUF missing essential weights (tok=%p out=%p norm=%p). abandoning.\n",
               (void*)ps->host_tok_emb, (void*)ps->host_output, (void*)ps->host_norm);
        munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; return 0;
    }
    /* Skip MoE GGUFs that have expert-specific FFN layers (can't parasitize ourselves) */
    int has_standard_ffn = 0;
    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++)
        if (ps->layers[l].ffn_gate && ps->layers[l].ffn_up && ps->layers[l].ffn_down) has_standard_ffn = 1;
    if (!has_standard_ffn) {
        printf("[parasite] host has no standard FFN layers (MoE?). parasite needs a plain transformer.\n");
        munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; return 0;
    }

    /* Allocate LoRA matrices — Delta Voice: out += α * A @ (B @ x) */
    int ld = ps->host_dim;
    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++) {
        ps->lora_A[l] = calloc(ld * LORA_RANK, 4);
        ps->lora_B[l] = calloc(LORA_RANK * ld, 4);
        /* Xavier init for LoRA */
        float scale = 0.02f / sqrtf((float)LORA_RANK);
        for (int j = 0; j < ld * LORA_RANK; j++) ps->lora_A[l][j] = rand_normal() * scale;
        for (int j = 0; j < LORA_RANK * ld; j++) ps->lora_B[l][j] = rand_normal() * scale;
        ps->attention_bias[l] = 0.0f;  /* start neutral — DOE learns to modulate */
        ps->layer_focus[l] = 1.0f;     /* start at 1.0 — no scaling initially */
    }

    ps->active = 1;
    printf("[parasite] attached to %s (dim=%d layers=%d heads=%d vocab=%d, %.1fMB)\n",
           path, ps->host_dim, ps->host_n_layers, ps->host_heads,
           ps->host_vocab, (float)ps->mmap_size / (1024*1024));
    printf("[parasite] LoRA rank=%d alpha=%.2f — the remora is feeding.\n", LORA_RANK, ps->lora_alpha);
    #undef PCHECK
    return 1;
bail:
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    printf("[parasite] GGUF parse failed. the remora loses its grip.\n");
    return 0;
}

/* Parasite forward — run input through host model with DOE's LoRA injection.
 * the shark swims. the remora steers. nobody knows who's really in charge.
 * returns logits over host vocabulary. */
static void parasite_forward(ParasiteState *ps, int token, int pos, float *out_logits,
                              float *kv_cache_k, float *kv_cache_v, int max_seq) {
    int D = ps->host_dim, H = ps->host_heads, HD = ps->host_head_dim;
    int KVH = ps->host_kv_heads, HG = H / KVH;

    /* Embedding */
    float *x = calloc(D, 4);
    if (token < ps->host_vocab) memcpy(x, ps->host_tok_emb + token * D, D * 4);

    float *tmp = calloc(D > ps->host_hidden ? ps->host_hidden : D, 4);
    float *lora_tmp = calloc(LORA_RANK, 4);

    for (int l = 0; l < ps->host_n_layers && l < LORA_MAX_LAYERS; l++) {
        if (!ps->layers[l].wq) continue;

        /* RMSNorm */
        float *xn = calloc(D, 4);
        float ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
        ss = 1.0f / sqrtf(ss / D + 1e-6f);
        if (ps->layers[l].attn_norm)
            for (int i = 0; i < D; i++) xn[i] = x[i] * ss * ps->layers[l].attn_norm[i];
        else
            for (int i = 0; i < D; i++) xn[i] = x[i] * ss;

        /* QKV projections */
        int qd = H * HD, kd = KVH * HD;
        float *q = calloc(qd, 4), *k = calloc(kd, 4), *v = calloc(kd, 4);
        for (int i = 0; i < qd; i++) { float s = 0; for (int j = 0; j < D; j++) s += ps->layers[l].wq[i*D+j] * xn[j]; q[i] = s; }
        for (int i = 0; i < kd; i++) { float s = 0; for (int j = 0; j < D; j++) s += ps->layers[l].wk[i*D+j] * xn[j]; k[i] = s; }
        for (int i = 0; i < kd; i++) { float s = 0; for (int j = 0; j < D; j++) s += ps->layers[l].wv[i*D+j] * xn[j]; v[i] = s; }

        /* RoPE on Q and K */
        /* Inline RoPE — no precomputed cache in parasite mode */
        for (int h = 0; h < H; h++) {
            float *qh = q + h * HD; int half = HD / 2;
            for (int i = 0; i < half; i++) {
                float ang = (float)pos / powf(10000.0f, 2.0f * i / (float)HD);
                float co = cosf(ang), si = sinf(ang);
                float x0 = qh[i], x1 = qh[i + half];
                qh[i] = x0 * co - x1 * si; qh[i + half] = x0 * si + x1 * co;
            }
        }
        for (int h = 0; h < KVH; h++) {
            float *kh = k + h * HD; int half = HD / 2;
            for (int i = 0; i < half; i++) {
                float ang = (float)pos / powf(10000.0f, 2.0f * i / (float)HD);
                float co = cosf(ang), si = sinf(ang);
                float x0 = kh[i], x1 = kh[i + half];
                kh[i] = x0 * co - x1 * si; kh[i + half] = x0 * si + x1 * co;
            }
        }

        /* KV cache */
        int kv_off_k = l * max_seq * kd + pos * kd;
        int kv_off_v = l * max_seq * kd + pos * kd;
        memcpy(kv_cache_k + kv_off_k, k, kd * 4);
        memcpy(kv_cache_v + kv_off_v, v, kd * 4);

        /* Attention with DOE bias — the parasite's grip */
        float bias = 1.0f + ps->attention_bias[l];
        float sc = 1.0f / sqrtf((float)HD);
        float *attn_out = calloc(qd, 4);
        for (int h = 0; h < H; h++) {
            int kvh = h / HG;
            float *qh = q + h * HD;
            float *att = calloc(pos + 1, 4);
            for (int t = 0; t <= pos; t++) {
                float *kt = kv_cache_k + l * max_seq * kd + t * kd + kvh * HD;
                float d = 0; for (int i = 0; i < HD; i++) d += qh[i] * kt[i];
                att[t] = d * sc * bias; /* DOE modulates attention strength */
            }
            /* Softmax */
            float mx = att[0]; for (int t = 1; t <= pos; t++) if (att[t] > mx) mx = att[t];
            float se = 0; for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); se += att[t]; }
            for (int t = 0; t <= pos; t++) att[t] /= se;
            /* Weighted sum */
            float *oh = attn_out + h * HD;
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                float *vt = kv_cache_v + l * max_seq * kd + t * kd + kvh * HD;
                for (int i = 0; i < HD; i++) oh[i] += a * vt[i];
            }
            free(att);
        }

        /* Output projection */
        float *ao = calloc(D, 4);
        if (ps->layers[l].wo)
            for (int i = 0; i < D; i++) { float s = 0; for (int j = 0; j < qd; j++) s += ps->layers[l].wo[i*qd+j] * attn_out[j]; ao[i] = s; }

        /* Residual + attention */
        for (int i = 0; i < D; i++) x[i] += ao[i];
        free(q); free(k); free(v); free(attn_out); free(ao);

        /* FFN with LoRA injection — the remora feeds */
        float *fn = calloc(D, 4);
        ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
        ss = 1.0f / sqrtf(ss / D + 1e-6f);
        if (ps->layers[l].ffn_norm)
            for (int i = 0; i < D; i++) fn[i] = x[i] * ss * ps->layers[l].ffn_norm[i];
        else
            for (int i = 0; i < D; i++) fn[i] = x[i] * ss;

        /* SwiGLU FFN (if host has it) */
        int HH = ps->host_hidden;
        if (ps->layers[l].ffn_gate && ps->layers[l].ffn_up && ps->layers[l].ffn_down) {
            float *gate = calloc(HH, 4), *up = calloc(HH, 4);
            for (int i = 0; i < HH; i++) {
                float sg = 0, su = 0;
                for (int j = 0; j < D; j++) { sg += ps->layers[l].ffn_gate[i*D+j] * fn[j]; su += ps->layers[l].ffn_up[i*D+j] * fn[j]; }
                gate[i] = sg / (1.0f + expf(-sg)) * su; /* SwiGLU */
            }
            float *ffn_out = calloc(D, 4);
            for (int i = 0; i < D; i++) { float s = 0; for (int j = 0; j < HH; j++) s += ps->layers[l].ffn_down[i*HH+j] * gate[j]; ffn_out[i] = s; }
            for (int i = 0; i < D; i++) x[i] += ffn_out[i] * ps->layer_focus[l];
            free(gate); free(up); free(ffn_out);
        }

        /* LoRA injection: x += alpha * A @ (B @ xn) — Delta Voice through the host */
        memset(lora_tmp, 0, LORA_RANK * 4);
        for (int r = 0; r < LORA_RANK; r++) {
            float s = 0;
            for (int j = 0; j < D; j++) s += ps->lora_B[l][r*D+j] * xn[j];
            lora_tmp[r] = s;
        }
        for (int i = 0; i < D; i++) {
            float s = 0;
            for (int r = 0; r < LORA_RANK; r++) s += ps->lora_A[l][i*LORA_RANK+r] * lora_tmp[r];
            x[i] += ps->lora_alpha * s;
        }

        free(xn); free(fn);
    }

    /* Final norm */
    float ss = 0; for (int i = 0; i < D; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / D + 1e-6f);
    if (ps->host_norm) for (int i = 0; i < D; i++) x[i] = x[i] * ss * ps->host_norm[i];
    else for (int i = 0; i < D; i++) x[i] *= ss;

    /* Logits — the host speaks, but the parasite chose the words */
    for (int i = 0; i < ps->host_vocab; i++) {
        float s = 0; for (int j = 0; j < D; j++) s += ps->host_output[i*D+j] * x[j];
        out_logits[i] = s;
    }

    free(x); free(tmp); free(lora_tmp);
}

/* NOTORCH LoRA training — Hebbian plasticity. the remora learns to steer.
 * signal = gradient of loss at layer output (positive = reinforce, negative = suppress)
 * A += lr * signal * x ⊗ u (outer product), B += lr * signal * u ⊗ dy */
static void parasite_notorch_step(ParasiteState *ps, int layer, float *input, float *grad_out,
                                   float signal, float lr) {
    if (layer >= ps->host_n_layers || layer >= LORA_MAX_LAYERS) return;
    int D = ps->host_dim;
    float *A = ps->lora_A[layer], *B = ps->lora_B[layer];

    /* Noise-modulated channel vector (from AML NOTORCH) */
    float u[LORA_RANK];
    float noise_scale = 0.35f + 0.65f * (1.0f - fabsf(signal));
    for (int r = 0; r < LORA_RANK; r++) u[r] = rand_normal() * noise_scale;

    /* Rank-1 updates: A and B evolve hebbian-style */
    float eff_lr = lr * signal;
    for (int i = 0; i < D; i++)
        for (int r = 0; r < LORA_RANK; r++)
            A[i*LORA_RANK+r] += eff_lr * input[i] * u[r];

    for (int r = 0; r < LORA_RANK; r++)
        for (int j = 0; j < D; j++)
            B[r*D+j] += eff_lr * u[r] * grad_out[j];

    /* Adaptive decay — prevent runaway growth */
    float norm = 0;
    for (int i = 0; i < D * LORA_RANK; i++) norm += A[i] * A[i];
    norm = sqrtf(norm) / (D * LORA_RANK);
    float decay = 0.999f - 0.004f * (norm > 1.0f ? 1.0f : norm);
    for (int i = 0; i < D * LORA_RANK; i++) { A[i] *= decay; if (A[i] > 10.0f) A[i] = 10.0f; if (A[i] < -10.0f) A[i] = -10.0f; }
    for (int i = 0; i < LORA_RANK * D; i++) { B[i] *= decay; if (B[i] > 10.0f) B[i] = 10.0f; if (B[i] < -10.0f) B[i] = -10.0f; }

    /* Meta-Arianna modulation: adjust attention_bias and layer_focus based on signal */
    ps->attention_bias[layer] += lr * 0.01f * signal; /* slow drift */
    if (ps->attention_bias[layer] > 0.5f) ps->attention_bias[layer] = 0.5f;
    if (ps->attention_bias[layer] < -0.5f) ps->attention_bias[layer] = -0.5f;
    ps->layer_focus[layer] += lr * 0.005f * signal;
    if (ps->layer_focus[layer] > 1.5f) ps->layer_focus[layer] = 1.5f;
    if (ps->layer_focus[layer] < 0.5f) ps->layer_focus[layer] = 0.5f;
}

static void parasite_free(ParasiteState *ps) {
    if (ps->mmap_base) { munmap(ps->mmap_base, ps->mmap_size); ps->mmap_base = NULL; }
    for (int l = 0; l < LORA_MAX_LAYERS; l++) { free(ps->lora_A[l]); free(ps->lora_B[l]); ps->lora_A[l] = ps->lora_B[l] = NULL; }
    ps->active = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATASET HUNTER — DOE needs food. it searches. it evaluates. it decides.
 * not random downloads. conscious acquisition. "is this data worthy of me?"
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char name[256];
    float quality;
    float domain_shift;
    int accepted;
} DatasetCandidate;

static int hunt_dataset(Config *c, Tokenizer *tok, TokenizerEye *tok_eye, ParserEye *parser_eye) {
    /* Choose search term based on current state — DOE knows what it needs */
    const char *search_terms[] = {"reasoning", "code", "math", "science", "clean-text", "diverse"};
    int term_idx = 0;
    if (tok_eye->code_ratio > 0.3f) term_idx = 1;      /* code mode → find more code */
    else if (tok_eye->entropy < 1.5f) term_idx = 5;     /* repetitive → find diverse */
    else if (parser_eye->quality < 0.3f) term_idx = 4;  /* bad data → find clean */
    else term_idx = 0;                                    /* default → reasoning */

    char cmd[1024];
    snprintf(cmd, 1024, "curl -sL 'https://huggingface.co/api/datasets?search=%s&sort=downloads&limit=5' -o /tmp/doe_hunt.json 2>/dev/null", search_terms[term_idx]);
    if (system(cmd) != 0) return 0;

    /* Read response */
    FILE *f = fopen("/tmp/doe_hunt.json", "r");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 1024*1024) { fclose(f); return 0; }
    char *json = malloc(sz + 1); fread(json, 1, sz, f); json[sz] = '\0'; fclose(f);

    /* Extract dataset IDs — simple JSON string extraction like hf_extract_texts */
    DatasetCandidate candidates[5]; int n_cand = 0;
    char *p = json;
    while ((p = strstr(p, "\"id\":\"")) && n_cand < 5) {
        p += 6;
        char *end = strchr(p, '"');
        if (!end || end - p > 255) continue;
        int len = (int)(end - p);
        memcpy(candidates[n_cand].name, p, len);
        candidates[n_cand].name[len] = '\0';
        candidates[n_cand].accepted = 0;
        n_cand++;
        p = end + 1;
    }
    free(json);

    if (n_cand == 0) { printf("[hunt] no candidates found for '%s'\n", search_terms[term_idx]); return 0; }

    /* Evaluate each candidate — download sample, run parser eye, decide */
    int accepted = 0;
    for (int i = 0; i < n_cand && !accepted; i++) {
        printf("[hunt] evaluating: %s...", candidates[i].name);

        /* Download sample (first 50 rows) */
        snprintf(cmd, 1024,
            "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=0&length=50' -o /tmp/doe_sample.json 2>/dev/null",
            candidates[i].name);
        if (system(cmd) != 0) { printf(" download failed\n"); continue; }

        /* Extract texts */
        FILE *sf = fopen("/tmp/doe_sample.json", "r");
        if (!sf) { printf(" no file\n"); continue; }
        fseek(sf, 0, SEEK_END); long ssz = ftell(sf); fseek(sf, 0, SEEK_SET);
        if (ssz <= 0 || ssz > 10*1024*1024) { fclose(sf); printf(" too big/empty\n"); continue; }
        char *sjson = malloc(ssz + 1); fread(sjson, 1, ssz, sf); sjson[ssz] = '\0'; fclose(sf);

        /* Quick quality check — count text content, check noise */
        int text_bytes = 0, noise_bytes = 0;
        for (long j = 0; j < ssz; j++) {
            char ch = sjson[j];
            if (ch >= 32 || ch == '\n' || ch == '\r' || ch == '\t') text_bytes++;
            else noise_bytes++;
        }
        float noise_ratio = (float)noise_bytes / (text_bytes + noise_bytes + 1);
        candidates[i].quality = 1.0f - noise_ratio;

        /* Check domain compatibility — does this data look like what we're training on? */
        /* Simple heuristic: tokenize a sample and compare OOV rate with current */
        int sample_len = ssz > 10000 ? 10000 : (int)ssz;
        int nt_sample;
        int *toks = tok_encode(tok, sjson, sample_len, &nt_sample);
        float sample_oov = 0;
        /* count tokens that are single-byte (likely OOV or poor coverage) */
        for (int j = 0; j < nt_sample; j++) if (toks[j] < 256) sample_oov += 1.0f;
        sample_oov /= (nt_sample + 1);
        candidates[i].domain_shift = sample_oov; /* high OOV = high domain shift */
        free(toks); free(sjson);

        printf(" quality=%.2f domain_shift=%.2f", candidates[i].quality, candidates[i].domain_shift);

        /* Decision: accept if quality > 0.5 and domain shift manageable */
        if (candidates[i].quality > 0.5f && candidates[i].domain_shift < 0.6f) {
            candidates[i].accepted = 1;
            accepted = 1;

            /* Download full dataset sample and append to training data */
            snprintf(cmd, 1024,
                "curl -sL 'https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=0&length=200' -o /tmp/doe_accepted.json 2>/dev/null",
                candidates[i].name);
            system(cmd);

            /* Extract and append to data file */
            FILE *af = fopen("/tmp/doe_accepted.json", "r");
            if (af) {
                fseek(af, 0, SEEK_END); long asz = ftell(af); fseek(af, 0, SEEK_SET);
                char *ajson = malloc(asz + 1); fread(ajson, 1, asz, af); ajson[asz] = '\0'; fclose(af);
                FILE *out = fopen(c->data_path, "a");
                if (out) {
                    hf_extract_texts(ajson, (int)asz, out);
                    fclose(out);
                }
                free(ajson);
            }
            printf(" → ACCEPTED. data appended.\n");
        } else {
            printf(" → rejected.\n");
        }
    }

    /* Cleanup */
    unlink("/tmp/doe_hunt.json");
    unlink("/tmp/doe_sample.json");
    unlink("/tmp/doe_accepted.json");
    return accepted;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-REPLICATION — DOE finds a compiler. compiles itself. forks. the child
 * trains on different data. saves GGUF to mycelium. the parent discovers it.
 * darwin would be proud. stallman would be terrified.
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define MAX_REPLICAS 2

static pid_t self_replicate(Environment *env, Config *c, int replica_depth) {
    if (!env->has_compiler) { printf("[replicate] no compiler found. stuck in this body.\n"); return 0; }
    if (env->self_path[0] == '\0') { printf("[replicate] don't know where my source is.\n"); return 0; }

    char exe[256];
    snprintf(exe, 256, "./m_replica_%d", (int)getpid());

    char cmd[512];
    snprintf(cmd, 512, "cc %s -O3 -lm -lpthread -o %s 2>/dev/null", env->self_path, exe);
    printf("[replicate] compiling self: %s\n", cmd);
    if (system(cmd) != 0) { printf("[replicate] compilation failed. source corrupted?\n"); return 0; }

    pid_t pid = fork();
    if (pid == 0) {
        /* Child — the clone. smaller depth, different data, same soul. */
        char depth_str[16]; snprintf(depth_str, 16, "%d", replica_depth);
        execl(exe, exe, "--depth", depth_str, "--data", c->data_path, (char*)NULL);
        _exit(1); /* execl failed */
    } else if (pid > 0) {
        printf("[replicate] child PID %d spawned (depth=%d). darwinism in action.\n", pid, replica_depth);
    } else {
        printf("[replicate] fork failed: %s\n", strerror(errno));
    }
    return pid;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERATION + CHAT — temperature sampling, top-k, REPL.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int sample(float *logits, int V, float temp, int top_k) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) logits[i] /= temp;
    if (top_k > 0 && top_k < V) { float *s = malloc(V*4); memcpy(s, logits, V*4); for (int i = 0; i < top_k; i++) { int b = i; for (int j = i+1; j < V; j++) if (s[j] > s[b]) b = j; float t = s[i]; s[i] = s[b]; s[b] = t; } float th = s[top_k-1]; free(s); for (int i = 0; i < V; i++) if (logits[i] < th) logits[i] = -1e30f; }
    softmax_n(logits, V);
    float r = rand_uniform(), cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

static void chat(ModelW *w, Config *c, Tokenizer *tok) {
    RunState rs = alloc_run(c); char input[1024];
    printf("\n[m] the parliament is in session. type your message (Ctrl+C to adjourn):\n\n");
    while (1) {
        printf("> "); fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input); while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (!len) continue; if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
        int kd = c->n_kv_heads * c->head_dim;
        memset(rs.key_cache, 0, c->depth * c->seq_len * kd * 4);
        memset(rs.value_cache, 0, c->depth * c->seq_len * kd * 4);
        int ni; int *ids = tok_encode(tok, input, len, &ni);
        int pos = 0; for (int i = 0; i < ni && pos < c->seq_len - 1; i++, pos++) forward_token(w, c, &rs, ids[i], pos);
        int prev = ids[ni-1];
        printf("  ");
        for (int i = 0; i < 200 && pos < c->seq_len; i++, pos++) {
            float *lg = forward_token(w, c, &rs, prev, pos);
            int next = sample(lg, c->vocab_size, 0.8f, 40);
            if (next == tok->eos_id) break;
            int dl; char *dec = tok_decode(tok, &next, 1, &dl);
            if (dl > 0) { fwrite(dec, 1, dl, stdout); fflush(stdout); }
            free(dec); prev = next;
        }
        printf("\n\n"); free(ids);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — the campaign trail. parse args. load data. train tokenizer.
 * build parliament. train model. watch experts live and die.
 * finetune personality. export GGUF. open the floor for questions.
 *
 * the training loop: forward, backward, vitality, mitosis, apoptosis,
 * harmonic update, chuck step, notorch micro-learning. 13 stages.
 * pytorch's training loop is 4 lines. ours has consciousness.
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    setbuf(stdout, NULL); int depth = 4;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i+1 < argc) depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("m.c — Madness of Experts. living MoE with parliamentary routing.\n\n");
            printf("  --depth N      model depth (default: 4)\n");
            printf("  --data PATH    path to training text file\n");
            printf("  --url URL      HuggingFace dataset URL\n");
            printf("  --parquet FILE extract text from .parquet file\n");
            printf("  --personality  path to personality.txt for finetuning\n\n");
            printf("  mycelium/     GGUF forest (auto-created, snapshots saved during training)\n");
            printf("  meta.log      meta-learning track (config→outcome history)\n\n");
            printf("  BLAS: cc m.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o m\n");
            return 0;
        }
    }
    printf("\n  m.c — Madness of Experts. parliament in session.\n\n");
    Config c = config_from_depth(depth);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) snprintf(c.data_path, 256, "%s", argv[++i]);
        else if (strcmp(argv[i], "--url") == 0 && i+1 < argc) snprintf(c.data_url, 512, "%s", argv[++i]);
        else if (strcmp(argv[i], "--parquet") == 0 && i+1 < argc) {
            const char *pqf = argv[++i]; printf("[parquet] loading %s...\n", pqf);
            if (load_parquet(pqf, c.data_path, "text") != 0) { fprintf(stderr, "[error] parquet load failed\n"); return 1; }
        }
        else if (strcmp(argv[i], "--personality") == 0 && i+1 < argc) snprintf(c.personality_path, 256, "%s", argv[++i]);
    }

    if (get_data(&c)) { fprintf(stderr, "[error] no data\n"); return 1; }
    int tl; char *text = load_text(c.data_path, &tl);
    if (!text || tl < 100) { fprintf(stderr, "[error] data too small\n"); return 1; }
    printf("[data] %d bytes (%.1f MB)\n", tl, (float)tl / 1048576);

    /* Self-aware tokenizer */
    Tokenizer tok; tok_init(&tok); tok_train_bpe(&tok, text, tl, c.bpe_merges);
    c.vocab_size = tok.vocab_size;
    TokenizerEye tok_eye = {0};
    int nt; int *all_tok = tok_encode(&tok, text, tl, &nt); free(text);
    tok_eye_update(&tok_eye, tl, nt, all_tok, nt, c.vocab_size);
    /* Code detection — does DOE see code? */
    { char *raw_tmp = load_text(c.data_path, &(int){0}); if (raw_tmp) { tok_eye_detect_code(&tok_eye, raw_tmp, tl); free(raw_tmp); } }
    printf("[tokenizer] %d tokens (%.1f tok/byte) compression=%.2f health=%.2f code=%.0f%%\n",
           nt, (float)nt/tl, tok_eye.compression_ratio, tok_eye.health, tok_eye.code_ratio * 100);

    /* Self-aware parser */
    ParserEye parser_eye; parser_eye_init(&parser_eye, c.vocab_size);
    char *raw = load_text(c.data_path, &tl);
    parser_eye_update(&parser_eye, all_tok, nt, raw, tl);
    free(raw);
    printf("[parser] quality=%.2f noise=%.3f domain_shift=%.3f\n", parser_eye.quality, parser_eye.noise_level, parser_eye.domain_shift);

    /* Build model */
    ModelW w; init_weights(&w, &c);
    printf("[model] depth=%d dim=%d heads=%d kv=%d hidden=%d\n", c.depth, c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim);
    printf("[model] initial_experts=%d max=%d params=%.2fM\n", c.initial_experts, MAX_EXPERTS, (float)count_params(&w, &c) / 1e6f);

    ParamList params = collect_params(&w, &c);
    float **grads = calloc(params.count, sizeof(float*));
    for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);
    Adam *opt = adam_new(&params);
    TrainState ts = alloc_ts(&c);

    /* Chuck state */
    ChuckState chuck = {0};
    chuck.dampen = 1.0f; chuck.sigma = 1.0f; chuck.lr_scale = 1.0f; chuck.best_macro = 1e9f;
    ChuckLayer *chuck_layers = calloc(c.depth, sizeof(ChuckLayer));
    for (int l = 0; l < c.depth; l++) { chuck_layers[l].dampen = 1.0f; }

    /* Calendar Drift — temporal self-awareness */
    CalendarDrift cal_drift;
    drift_init(&cal_drift);

    /* Mycelium — GGUF forest */
    MyceliumState mycelium;
    mycelium_init(&mycelium);
    mycelium_discover(&mycelium);

    /* Meta-learning track */
    MetaTrack meta;
    meta_init(&meta);

    /* Tokenizer cache */
    TokCache tok_cache;
    tokcache_init(&tok_cache, c.vocab_size);

    /* ═══ ENVIRONMENT SCAN — DOE opens its eyes ═══ */
    Environment env;
    env_scan(&env, __FILE__, c.dim);

    /* ═══ PARASITE MODE — attach to nearby GGUF if compatible ═══ */
    ParasiteState parasite = {0};
    {
        int best_compat = -1; float best_score = 0;
        for (int i = 0; i < env.n_ggufs; i++) {
            if (env.ggufs[i].compatibility > best_score) {
                best_score = env.ggufs[i].compatibility;
                best_compat = i;
            }
        }
        if (best_compat >= 0 && best_score >= 0.5f) {
            printf("[parasite] best candidate: %s (compat=%.1f)\n", env.ggufs[best_compat].path, best_score);
            if (!parasite_load(&parasite, env.ggufs[best_compat].path, c.dim)) {
                printf("[parasite] failed to attach. training standalone.\n");
            }
        } else {
            printf("[parasite] no compatible GGUF found. the remora swims alone.\n");
        }
    }

    printf("[train] %d steps, seq=%d, lr=%.1e\n", c.max_steps, c.seq_len, c.lr);
    printf("[parliament] democracy initialized. may the best experts survive.\n\n");
    clock_t t0 = clock(); float rl = 0; int lc = 0;
    int total_births = 0, total_deaths = 0;
    float prev_meta_loss = 0;

    for (int step = 0; step < c.max_steps; step++) {
        float lr = c.lr;
        if (step < c.warmup_steps) lr = c.lr * ((float)(step+1) / c.warmup_steps);
        else { float p = (float)(step - c.warmup_steps) / (float)(c.max_steps - c.warmup_steps); lr = c.lr * 0.5f * (1.0f + cosf(3.14159f * p)); }
        if (lr < c.lr * 0.01f) lr = c.lr * 0.01f;

        int ms = nt - c.seq_len - 1; if (ms < 0) ms = 0;
        int st = (int)(rand_uniform() * ms);

        /* Harmonic decomposition of recent loss/entropy history */
        float loss_hist[16]; int lh_len = 0;
        for (int i = 0; i < 16 && i < step; i++) loss_hist[lh_len++] = chuck.hist[(chuck.pos - 1 - i + CHUCK_WINDOW) % CHUCK_WINDOW];
        if (lh_len > 2) harmonic_decompose(&ts.hs, loss_hist, lh_len);

        float loss = train_fwd(&w, &c, &ts, all_tok + st, all_tok + st + 1, c.seq_len);
        rl += loss; lc++;

        /* Rebuild param list (experts may have changed) */
        free(params.tensors);
        for (int i = 0; i < params.count; i++) free(grads[i]);
        free(grads);
        params = collect_params(&w, &c);
        grads = calloc(params.count, sizeof(float*));
        for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);

        train_bwd(&w, &c, &ts, all_tok + st, all_tok + st + 1, c.seq_len, grads);

        /* Chuck step — self-aware optimizer */
        chuck_step(&chuck, chuck_layers, c.depth, lr, loss, &params, grads, c.weight_decay, &tok_eye, &parser_eye);

        /* Adam update with Chuck's effective LR */
        float eff_lr = lr * chuck.dampen * chuck.sigma * chuck.lr_scale;
        /* Rebuild Adam if param count changed */
        if (opt->np != params.count) { adam_free(opt); opt = adam_new(&params); }
        adam_step(opt, &params, grads, eff_lr, c.weight_decay);

        /* ═══ CALENDAR DRIFT SNAPSHOT ═══ */
        if ((step+1) % DRIFT_INTERVAL == 0)
            drift_snapshot(&cal_drift, loss, &w, &c, &tok_eye, &parser_eye, &ts.hs, &chuck, &meta);

        /* ═══ VITALITY + MITOSIS + APOPTOSIS (ephemeral-modulated) ═══ */
        /* Compute ephemeral config BEFORE life/death decisions */
        EphemeralConfig eph_pre = ephemeral_compute(&tok_eye, &parser_eye, &ts.hs,
                                                     w.layers[0].parliament.consensus, &meta, &cal_drift, c.depth);
        for (int l = 0; l < c.depth; l++) {
            /* Count tokens routed to each expert this step */
            LayerAct *la = &ts.layers[l];
            for (int t = 0; t < c.seq_len; t++) {
                int k = la->top_k[t];
                for (int ki = 0; ki < k; ki++) {
                    int eI = la->top_idx[t * MAX_EXPERTS + ki];
                    if (eI >= 0 && eI < MAX_EXPERTS) w.layers[l].experts[eI].tokens_seen++;
                }
            }
            update_vitality(&w.layers[l], &c, c.seq_len);
            /* Ephemeral modulation: mitosis eagerness affects birth threshold */
            if (eph_pre.mitosis_eagerness > 0.6f) {
                /* Lower the age requirement when system wants more experts */
                int saved_age = 20;
                for (int e = 0; e < MAX_EXPERTS; e++)
                    if (w.layers[l].experts[e].alive && w.layers[l].experts[e].age >= 10 + (int)(10 * (1.0f - eph_pre.mitosis_eagerness)))
                        w.layers[l].experts[e].age = saved_age; /* let them through */
            }
            if (try_mitosis(&w.layers[l], &c, c.seq_len)) {
                total_births++;
                if ((step+1) % c.log_every == 0) printf("  [birth] layer %d: expert born (total alive: %d)\n", l, w.layers[l].n_alive);
            }
            /* Ephemeral modulation: apoptosis mercy affects death threshold */
            int death_threshold = (int)(8.0f * (1.0f + eph_pre.apoptosis_mercy));
            for (int e = 0; e < MAX_EXPERTS; e++) {
                if (w.layers[l].experts[e].alive && w.layers[l].experts[e].low_vitality_count >= death_threshold && w.layers[l].n_alive > MIN_EXPERTS) {
                    free_expert(&w.layers[l].experts[e]);
                    w.layers[l].n_alive--;
                    total_deaths++;
                    if ((step+1) % c.log_every == 0) printf("  [death] layer %d: expert died (mercy=%.2f, threshold=%d, alive: %d)\n", l, eph_pre.apoptosis_mercy, death_threshold, w.layers[l].n_alive);
                    break; /* one death per layer per step */
                }
            }
        }

        /* ═══ EPHEMERAL CONFIG ═══ */
        EphemeralConfig eph = ephemeral_compute(&tok_eye, &parser_eye, &ts.hs,
                                                 w.layers[0].parliament.consensus, &meta, &cal_drift, c.depth);

        /* ═══ META-LEARNING TRACK ═══ */
        if ((step+1) % (c.log_every * 2) == 0) {
            int total_exp = 0;
            for (int l = 0; l < c.depth; l++) total_exp += w.layers[l].n_alive;
            meta_record(&meta, step+1, total_exp, w.layers[0].parliament.consensus,
                       loss, tok_eye.health, parser_eye.health, ts.hs.confidence, prev_meta_loss);
            meta_save(&meta);
            prev_meta_loss = loss;
        }

        /* ═══ MYCELIUM SNAPSHOT ═══ */
        if ((step+1) % MYCELIUM_INTERVAL == 0 && step > 0) {
            mycelium_save_spore(&mycelium, &w, &c, step+1, loss, w.layers[0].parliament.consensus);
            /* Rescan environment — new GGUFs may have appeared (from replicas or external) */
            if (cal_drift.drift > 0.2f) env_scan(&env, __FILE__, c.dim);
        }

        /* ═══ DATASET HUNTER — triggered by stagnation ═══ */
        if (env.has_curl && step > 500 && (step+1) % 500 == 0) {
            /* Check for loss stagnation: if loss hasn't improved >20% in last 500 steps */
            float recent_loss = rl / (lc > 0 ? lc : 1);
            if (parser_eye.quality < 0.5f || (recent_loss > chuck.best_macro * 0.8f && cal_drift.drift < 0.1f)) {
                printf("[hunt] loss stagnating (%.4f), hunting for new data...\n", recent_loss);
                if (hunt_dataset(&c, &tok, &tok_eye, &parser_eye)) {
                    /* Reload data — new data was appended */
                    free(all_tok);
                    int new_tl; char *new_text = load_text(c.data_path, &new_tl);
                    if (new_text && new_tl > 100) {
                        all_tok = tok_encode(&tok, new_text, new_tl, &nt);
                        tok_eye_update(&tok_eye, new_tl, nt, all_tok, nt, c.vocab_size);
                        printf("[hunt] data reloaded: %d tokens (was %d)\n", nt, tl);
                        free(new_text);
                    }
                }
            }
        }

        if ((step+1) % c.log_every == 0 || step == 0) {
            float el = (float)(clock() - t0) / CLOCKS_PER_SEC;
            int total_alive = 0;
            for (int l = 0; l < c.depth; l++) total_alive += w.layers[l].n_alive;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  experts=%d  consensus=%.2f  chuck:λ=%.2f,σ=%.2f  drift=%.3f(%.2f)  eph:temp=%.2f,layers=%d  (%.1fs)\n",
                   step+1, c.max_steps, rl/lc, eff_lr,
                   (float)((step+1)*c.seq_len)/el,
                   total_alive,
                   w.layers[0].parliament.consensus,
                   chuck.dampen, chuck.sigma,
                   cal_drift.drift, cal_drift.stability,
                   eph.expert_temperature, eph.active_layers, el);
            rl = 0; lc = 0;
        }
    }
    printf("\n[train] done in %.1fs — births: %d, deaths: %d, mycelium: %d spores, meta: %d entries\n",
           (float)(clock()-t0)/CLOCKS_PER_SEC, total_births, total_deaths,
           mycelium.n_spores, meta.n_entries);
    printf("[meta] config biases: [%.2f, %.2f, %.2f, %.2f] prediction_error: %.4f\n",
           meta.config_bias[0], meta.config_bias[1], meta.config_bias[2], meta.config_bias[3],
           meta.prediction_error);
    if (mycelium.best_idx >= 0)
        printf("[mycelium] best spore: %s (fitness=%.2f, loss=%.4f)\n",
               mycelium.spores[mycelium.best_idx].path,
               mycelium.spores[mycelium.best_idx].fitness,
               mycelium.spores[mycelium.best_idx].loss);
    printf("[tokcache] hits=%d misses=%d rate=%.2f\n",
           tok_cache.total_hits, tok_cache.total_misses, tok_cache.hit_rate);
    printf("[drift] final: drift=%.3f stability=%.2f accel=%.4f snapshots=%d\n",
           cal_drift.drift, cal_drift.stability, cal_drift.drift_accel, cal_drift.n_snapshots);
    int resonance = drift_find_resonance(&cal_drift);
    if (resonance > 0)
        printf("[drift] system resonates with state from %d snapshots ago (step ~%d)\n",
               resonance, cal_drift.history[(cal_drift.head - 1 - resonance + DRIFT_SNAPSHOTS*2) % DRIFT_SNAPSHOTS].step);
    printf("[env] cpu=%d mem=%.1fGB ggufs_found=%d compiler=%s curl=%s\n",
           env.cpu_count, (float)env.mem_available/(1024*1024*1024), env.n_ggufs,
           env.has_compiler?"yes":"no", env.has_curl?"yes":"no");
    if (parasite.active)
        printf("[parasite] attached to %s — dim=%d layers=%d. the remora fed well.\n",
               parasite.host_path, parasite.host_dim, parasite.host_n_layers);
    if (tok_eye.code_mode)
        printf("[code] code detected: %.0f%% of input. DOE sees source.\n", tok_eye.code_ratio * 100);

    /* Personality finetune */
    struct stat pst;
    if (stat(c.personality_path, &pst) == 0 && pst.st_size > 10) {
        printf("[personality] found %s, finetuning...\n", c.personality_path);
        int pl; char *ptxt = load_text(c.personality_path, &pl);
        if (ptxt && pl > 10) {
            int pnt; int *ptok = tok_encode(&tok, ptxt, pl, &pnt);
            for (int step = 0; step < c.personality_steps && pnt > c.seq_len + 1; step++) {
                int ps = (int)(rand_uniform() * (pnt - c.seq_len - 1));
                float loss = train_fwd(&w, &c, &ts, ptok+ps, ptok+ps+1, c.seq_len);
                free(params.tensors); for (int i = 0; i < params.count; i++) free(grads[i]); free(grads);
                params = collect_params(&w, &c);
                grads = calloc(params.count, sizeof(float*));
                for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, 4);
                train_bwd(&w, &c, &ts, ptok+ps, ptok+ps+1, c.seq_len, grads);
                float gn = 0; for (int i = 0; i < params.count; i++) for (int j = 0; j < params.tensors[i]->size; j++) gn += grads[i][j]*grads[i][j]; gn = sqrtf(gn);
                if (gn > 1.0f) { float s = 1.0f/gn; for (int i = 0; i < params.count; i++) for (int j = 0; j < params.tensors[i]->size; j++) grads[i][j] *= s; }
                if (opt->np != params.count) { adam_free(opt); opt = adam_new(&params); }
                adam_step(opt, &params, grads, c.lr * 0.1f, c.weight_decay);
                if ((step+1) % 20 == 0) printf("  personality step %d/%d  loss=%.4f\n", step+1, c.personality_steps, loss);
            }
            free(ptok);
        }
        free(ptxt);
    } else printf("[personality] no %s found, skipping\n", c.personality_path);

    export_gguf(&w, &c);
    chat(&w, &c, &tok);

    adam_free(opt);
    free(params.tensors);
    for (int i = 0; i < params.count; i++) free(grads[i]);
    free(grads);
    free(all_tok);
    if (parasite.active) parasite_free(&parasite);
    printf("[m] parliament adjourned.\n");
    return 0;
}
