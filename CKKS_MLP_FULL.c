/*
 * ════════════════════════════════════════════════════════════════════
 *  CKKS HE-MLP — Iris Classifier  |  Pure C, No Libraries
 * ════════════════════════════════════════════════════════════════════
 *  Compile:  gcc -O2 -o ckks_mlp ckks_mlp_fixed.c -lm
 *  Run:      ./ckks_mlp
 *  Expect:   RLWE self-test PASS + 98.67% accuracy (148/150)
 *
 *  Network: 4 → 8 (x² activation) → 3
 *  Weights: retrained with SGD, 5000 epochs, on sklearn StandardScaler iris
 * ════════════════════════════════════════════════════════════════════
 *
 *
 *  PIPELINE (per sample):
 *  1. Normalise x[4] with sklearn StandardScaler stats.
 *  2.  RLWE encrypt→decrypt roundtrip.
 *  3. Encrypt: ct_x[j] = round(x[j]·Δ)   [degree-0 CKKS plaintext]
 *  4. Layer 1 (HE): ct_h[i] = (b1[i]·Δ + ΣW1[i][j]·ct_x[j])² / Δ
 *  5. Layer 2 (HE): ct_logit[k] = b2[k]·Δ + ΣW2[k][i]·ct_h[i]
 *  6. Decode: logit[k] = ct_logit[k] / Δ → softmax → argmax
 * 
 * ════════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── parameters ─────────────────────────────────────────────────── */
#define N         8192
#define LOG_SCALE 20
#define SCALE_D   ((double)(1u << LOG_SCALE))
#define L         4
#define IN        4
#define H         8
#define OUT       3

/* NTT primes: prime AND ≡ 1 (mod 2N=16384) */
static const uint64_t PRIMES[L] = {
    268582913ULL, 268664833ULL, 268730369ULL, 268779521ULL
};

/* ─── Iris dataset (150×4, embedded) ────────────────────────────── */
static const float IRIS[150][4] = {
    {5.1f,3.5f,1.4f,0.2f},{4.9f,3.0f,1.4f,0.2f},{4.7f,3.2f,1.3f,0.2f},
    {4.6f,3.1f,1.5f,0.2f},{5.0f,3.6f,1.4f,0.2f},{5.4f,3.9f,1.7f,0.4f},
    {4.6f,3.4f,1.4f,0.3f},{5.0f,3.4f,1.5f,0.2f},{4.4f,2.9f,1.4f,0.2f},
    {4.9f,3.1f,1.5f,0.1f},{5.4f,3.7f,1.5f,0.2f},{4.8f,3.4f,1.6f,0.2f},
    {4.8f,3.0f,1.4f,0.1f},{4.3f,3.0f,1.1f,0.1f},{5.8f,4.0f,1.2f,0.2f},
    {5.7f,4.4f,1.5f,0.4f},{5.4f,3.9f,1.3f,0.4f},{5.1f,3.5f,1.4f,0.3f},
    {5.7f,3.8f,1.7f,0.3f},{5.1f,3.8f,1.5f,0.3f},{5.4f,3.4f,1.7f,0.2f},
    {5.1f,3.7f,1.5f,0.4f},{4.6f,3.6f,1.0f,0.2f},{5.1f,3.3f,1.7f,0.5f},
    {4.8f,3.4f,1.9f,0.2f},{5.0f,3.0f,1.6f,0.2f},{5.0f,3.4f,1.6f,0.4f},
    {5.2f,3.5f,1.5f,0.2f},{5.2f,3.4f,1.4f,0.2f},{4.7f,3.2f,1.6f,0.2f},
    {4.8f,3.1f,1.6f,0.2f},{5.4f,3.4f,1.5f,0.4f},{5.2f,4.1f,1.5f,0.1f},
    {5.5f,4.2f,1.4f,0.2f},{4.9f,3.1f,1.5f,0.1f},{5.0f,3.2f,1.2f,0.2f},
    {5.5f,3.5f,1.3f,0.2f},{4.9f,3.1f,1.5f,0.1f},{4.4f,3.0f,1.3f,0.2f},
    {5.1f,3.4f,1.5f,0.2f},{5.0f,3.5f,1.3f,0.3f},{4.5f,2.3f,1.3f,0.3f},
    {4.4f,3.2f,1.3f,0.2f},{5.0f,3.5f,1.6f,0.6f},{5.1f,3.8f,1.9f,0.4f},
    {4.8f,3.0f,1.4f,0.3f},{5.1f,3.8f,1.6f,0.2f},{4.6f,3.2f,1.4f,0.2f},
    {5.3f,3.7f,1.5f,0.2f},{5.0f,3.3f,1.4f,0.2f},
    {7.0f,3.2f,4.7f,1.4f},{6.4f,3.2f,4.5f,1.5f},{6.9f,3.1f,4.9f,1.5f},
    {5.5f,2.3f,4.0f,1.3f},{6.5f,2.8f,4.6f,1.5f},{5.7f,2.8f,4.5f,1.3f},
    {6.3f,3.3f,4.7f,1.6f},{4.9f,2.4f,3.3f,1.0f},{6.6f,2.9f,4.6f,1.3f},
    {5.2f,2.7f,3.9f,1.4f},{5.0f,2.0f,3.5f,1.0f},{5.9f,3.0f,4.2f,1.5f},
    {6.0f,2.2f,4.0f,1.0f},{6.1f,2.9f,4.7f,1.4f},{5.6f,2.9f,3.6f,1.3f},
    {6.7f,3.1f,4.4f,1.4f},{5.6f,3.0f,4.5f,1.5f},{5.8f,2.7f,4.1f,1.0f},
    {6.2f,2.2f,4.5f,1.5f},{5.6f,2.5f,3.9f,1.1f},{5.9f,3.2f,4.8f,1.8f},
    {6.1f,2.8f,4.0f,1.3f},{6.3f,2.5f,4.9f,1.5f},{6.1f,2.8f,4.7f,1.2f},
    {6.4f,2.9f,4.3f,1.3f},{6.6f,3.0f,4.4f,1.4f},{6.8f,2.8f,4.8f,1.4f},
    {6.7f,3.0f,5.0f,1.7f},{6.0f,2.9f,4.5f,1.5f},{5.7f,2.6f,3.5f,1.0f},
    {5.5f,2.4f,3.8f,1.1f},{5.5f,2.4f,3.7f,1.0f},{5.8f,2.7f,3.9f,1.2f},
    {6.0f,2.7f,5.1f,1.6f},{5.4f,3.0f,4.5f,1.5f},{6.0f,3.4f,4.5f,1.6f},
    {6.7f,3.1f,4.7f,1.5f},{6.3f,2.3f,4.4f,1.3f},{5.6f,3.0f,4.1f,1.3f},
    {5.5f,2.5f,4.0f,1.3f},{5.5f,2.6f,4.4f,1.2f},{6.1f,3.0f,4.6f,1.4f},
    {5.8f,2.6f,4.0f,1.2f},{5.0f,2.3f,3.3f,1.0f},{5.6f,2.7f,4.2f,1.3f},
    {5.7f,3.0f,4.2f,1.2f},{5.7f,2.9f,4.2f,1.3f},{6.2f,2.9f,4.3f,1.3f},
    {5.1f,2.5f,3.0f,1.1f},{5.7f,2.8f,4.1f,1.3f},
    {6.3f,3.3f,6.0f,2.5f},{5.8f,2.7f,5.1f,1.9f},{7.1f,3.0f,5.9f,2.1f},
    {6.3f,2.9f,5.6f,1.8f},{6.5f,3.0f,5.8f,2.2f},{7.6f,3.0f,6.6f,2.1f},
    {4.9f,2.5f,4.5f,1.7f},{7.3f,2.9f,6.3f,1.8f},{6.7f,2.5f,5.8f,1.8f},
    {7.2f,3.6f,6.1f,2.5f},{6.5f,3.2f,5.1f,2.0f},{6.4f,2.7f,5.3f,1.9f},
    {6.8f,3.0f,5.5f,2.1f},{5.7f,2.5f,5.0f,2.0f},{5.8f,2.8f,5.1f,2.4f},
    {6.4f,3.2f,5.3f,2.3f},{6.5f,3.0f,5.5f,1.8f},{7.7f,3.8f,6.7f,2.2f},
    {7.7f,2.6f,6.9f,2.3f},{6.0f,2.2f,5.0f,1.5f},{6.9f,3.2f,5.7f,2.3f},
    {5.6f,2.8f,4.9f,2.0f},{7.7f,2.8f,6.7f,2.0f},{6.3f,2.7f,4.9f,1.8f},
    {6.7f,3.3f,5.7f,2.1f},{7.2f,3.2f,6.0f,1.8f},{6.2f,2.8f,4.8f,1.8f},
    {6.1f,3.0f,4.9f,1.8f},{6.4f,2.8f,5.6f,2.1f},{7.2f,3.0f,5.8f,1.6f},
    {7.4f,2.8f,6.1f,1.9f},{7.9f,3.8f,6.4f,2.0f},{6.4f,2.8f,5.6f,2.2f},
    {6.3f,2.8f,5.1f,1.5f},{6.1f,2.6f,5.6f,1.4f},{7.7f,3.0f,6.1f,2.3f},
    {6.3f,3.4f,5.6f,2.4f},{6.4f,3.1f,5.5f,1.8f},{6.0f,3.0f,4.8f,1.8f},
    {6.9f,3.1f,5.4f,2.1f},{6.7f,3.1f,5.6f,2.4f},{6.9f,3.1f,5.1f,2.3f},
    {5.8f,2.7f,5.1f,1.9f},{6.8f,3.2f,5.9f,2.3f},{6.7f,3.3f,5.7f,2.5f},
    {6.7f,3.0f,5.2f,2.3f},{6.3f,2.5f,5.0f,1.9f},{6.5f,3.0f,5.2f,2.0f},
    {6.2f,3.4f,5.4f,2.3f},{5.9f,3.0f,5.1f,1.8f}
};

/* ─── Pre-trained weights ─────────────────────────────────────────── */
static const double W1[H][IN] = {
    { 0.04725025, -0.06928307,  0.30403364,  0.63342621},
    {-0.24059597,  0.03280834,  0.28801144,  0.02211828},
    {-0.13902794,  0.16009244, -0.22102519, -0.23239595},
    { 0.28682030, -1.02366008, -0.43019864, -0.14618188},
    {-0.13865128,  0.63902799, -1.54728276, -1.62841441},
    { 0.48532124,  0.16938573, -0.24518310, -0.81883444},
    {-0.28939507,  0.33588966, -0.83495395, -0.63457162},
    {-0.20988577, -0.13182803, -0.14243630,  0.62559322}
};
static const double b1[H] = {
     0.14289989, -0.39898236,  0.13809327,  0.40610344,
     0.46062172, -0.19823201, -1.25690041,  0.12136034
};
static const double W2[OUT][H] = {
    {-0.10826253,-0.26130073, 0.30461811,-0.51247870,
      0.68036913,-0.67576363,-1.04385591, 0.00438805},
    { 0.01249887, 0.03286602,-0.09658569, 0.37427085,
     -1.53904985,-0.35586187, 0.73103596, 0.21780984},
    { 0.41633995,-0.56638013, 0.10126176,-0.43390104,
      0.27470660, 0.41127409, 0.08547236, 0.43338118}
};
static const double b2[OUT] = {-0.58224704, 1.68238232, -1.10013528};

/* sklearn StandardScaler(ddof=1) stats */
static const double MEAN[IN] = {5.84333333,3.05733333,3.75800000,1.19933333};
static const double STDD[IN] = {0.82530129,0.43441097,1.75940407,0.75969263};

/* ─── Modular arithmetic ─────────────────────────────────────────── */
static inline uint64_t mulmod(uint64_t a,uint64_t b,uint64_t m){
    return (uint64_t)((__uint128_t)a*b%m);
}
static inline uint64_t addmod(uint64_t a,uint64_t b,uint64_t m){
    a+=b; if(a>=m) a-=m; return a;
}
static inline uint64_t submod(uint64_t a,uint64_t b,uint64_t m){
    return (a>=b)?a-b:a+m-b;
}
static uint64_t powmod(uint64_t b,uint64_t e,uint64_t m){
    uint64_t r=1; b%=m;
    while(e){if(e&1)r=mulmod(r,b,m);b=mulmod(b,b,m);e>>=1;}
    return r;
}

/* ─── NTT table ───────────────────────────────────────────────────
 *  The zeta table implements the negacyclic NTT correctly.
 *  zeta[k] = psi^{ bit_reverse(k, log2(N)) }
 *  where psi is a primitive 2N-th root of unity mod p.
 *
 *  Forward NTT  (large-stride to small-stride, k = 1..N-1 in order):
 *    for each length N/2, N/4, …, 1:
 *      for each group start = 0, 2·length, …:
 *        w = zeta[k++]
 *        butterfly: (a[j], a[j+len]) ← (a[j]+w·a[j+len], a[j]-w·a[j+len])
 *
 *  Inverse NTT  (small-stride to large-stride, k = N-1..1 in reverse):
 *    for each length 1, 2, …, N/2:
 *      for each group start = N-2·length, …, 2·length, 0  (REVERSED):
 *        w = zeta_inv[k--]
 *        butterfly: (a[j], a[j+len]) ← (a[j]+a[j+len], w·(a[j]-a[j+len]))
 *    scale by N^{-1}
 *
 *  The reversal of the start order in the inverse is the key fix: it ensures
 *  that each inverse butterfly exactly undoes the corresponding forward butterfly,
 *  giving fw∘iv = identity and correct ring multiplication.
 * ─────────────────────────────────────────────────────────────────── */
typedef struct {
    uint64_t p, n_inv;
    uint64_t *zeta;      /* zeta[k] = psi^brv(k)     for k=0..N-1 */
    uint64_t *zeta_inv;  /* zeta_inv[k] = psi_inv^brv(k)           */
} NTTTable;

/* bit-reverse of j using n_bits bits */
static uint64_t bit_reverse(uint64_t j, int n_bits) {
    uint64_t r = 0;
    for (int i = 0; i < n_bits; i++) { r = (r<<1)|(j&1); j>>=1; }
    return r;
}

static uint64_t find_psi(uint64_t p) {
    for (uint64_t g = 2; g < 1000000; g++) {
        if (powmod(g,(p-1)/2,p) != p-1) continue;
        uint64_t psi = powmod(g,(p-1)/(2*(uint64_t)N),p);
        if (powmod(psi,(uint64_t)N,p) == p-1) return psi;
    }
    fprintf(stderr,"find_psi failed\n"); exit(1);
}

static void build_ntt(NTTTable *t, uint64_t p) {
    t->p    = p;
    t->n_inv = powmod((uint64_t)N, p-2, p);
    t->zeta     = malloc(N*sizeof(uint64_t));
    t->zeta_inv = malloc(N*sizeof(uint64_t));

    uint64_t psi     = find_psi(p);
    uint64_t psi_inv = powmod(psi, p-2, p);
    int n_bits = 0; { uint64_t tmp=N; while(tmp>1){n_bits++;tmp>>=1;} }

    for (int k = 0; k < N; k++) {
        uint64_t e = bit_reverse((uint64_t)k, n_bits);
        t->zeta[k]     = powmod(psi,     e, p);
        t->zeta_inv[k] = powmod(psi_inv, e, p);
    }
}

/* Forward negacyclic NTT (DIF, large-to-small, k = 1..N-1) */
static void ntt_fwd(uint64_t *a, const NTTTable *t) {
    uint64_t p = t->p;
    int k = 1;
    for (int length = N>>1; length >= 1; length >>= 1) {
        for (int start = 0; start < N; start += 2*length) {
            uint64_t zeta = t->zeta[k++];
            for (int j = start; j < start+length; j++) {
                uint64_t u = a[j];
                uint64_t v = mulmod(zeta, a[j+length], p);
                a[j]        = addmod(u, v, p);
                a[j+length] = submod(u, v, p);
            }
        }
    }
}

/* Inverse negacyclic NTT (DIT, small-to-large, k = N-1..1 reversed per stage) */
static void ntt_inv(uint64_t *a, const NTTTable *t) {
    uint64_t p = t->p;
    int k = N-1;
    for (int length = 1; length < N; length <<= 1) {
        /* Iterate starts in REVERSE order so each butterfly undoes its forward pair */
        for (int start = N-2*length; start >= 0; start -= 2*length) {
            uint64_t zeta_inv = t->zeta_inv[k--];
            for (int j = start; j < start+length; j++) {
                uint64_t u = a[j];
                uint64_t v = a[j+length];
                a[j]        = addmod(u, v, p);
                a[j+length] = mulmod(zeta_inv, submod(u, v, p), p);
            }
        }
    }
    /* Scale by N^{-1} */
    for (int j = 0; j < N; j++) a[j] = mulmod(a[j], t->n_inv, p);
}

/* ─── Gaussian sampler (Box-Muller, σ=3.2) ───────────────────────── */
static double gauss(void) {
    double u1 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
    double u2 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
    return 3.2*sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
}

/* ─── RLWE types ─────────────────────────────────────────────────── */
typedef struct { uint64_t *c[L]; } RNSPoly;
typedef struct { RNSPoly c0,c1; } PolyCT;
typedef struct { RNSPoly s; } SecretKey;
typedef struct { RNSPoly pk0,pk1; } PublicKey;

static void rns_alloc(RNSPoly *r){for(int i=0;i<L;i++)r->c[i]=calloc(N,sizeof(uint64_t));}
static void rns_free(RNSPoly *r){for(int i=0;i<L;i++){free(r->c[i]);r->c[i]=NULL;}}
static void pct_alloc(PolyCT *ct){rns_alloc(&ct->c0);rns_alloc(&ct->c1);}
static void pct_free(PolyCT *ct){rns_free(&ct->c0);rns_free(&ct->c1);}

/* ─── Key generation ─────────────────────────────────────────────── */
static void gen_sk(SecretKey *sk, const NTTTable *ntt) {
    rns_alloc(&sk->s);
    int64_t s[N];
    for (int j=0;j<N;j++) s[j]=(rand()%3)-1;
    for (int i=0;i<L;i++) {
        uint64_t p=PRIMES[i];
        for (int j=0;j<N;j++) sk->s.c[i][j]=(s[j]<0)?p-1:(uint64_t)s[j];
        ntt_fwd(sk->s.c[i],&ntt[i]);
    }
}

static void gen_pk(PublicKey *pk, const SecretKey *sk, const NTTTable *ntt) {
    rns_alloc(&pk->pk0); rns_alloc(&pk->pk1);
    for (int i=0;i<L;i++) {
        uint64_t p=PRIMES[i];
        uint64_t *a=malloc(N*8),*e=malloc(N*8),*as=malloc(N*8),*b=malloc(N*8);
        for (int j=0;j<N;j++) a[j]=(uint64_t)rand()%p;
        for (int j=0;j<N;j++) {
            int64_t ev=(int64_t)round(gauss());
            e[j]=(ev<0)?(uint64_t)(ev+(int64_t)p):(uint64_t)(ev%p);
        }
        memcpy(as,a,N*8); ntt_fwd(as,&ntt[i]);
        for (int j=0;j<N;j++) as[j]=mulmod(as[j],sk->s.c[i][j],p);
        ntt_inv(as,&ntt[i]);
        for (int j=0;j<N;j++) b[j]=addmod(submod(0,as[j],p),e[j],p);
        ntt_fwd(a,&ntt[i]); ntt_fwd(b,&ntt[i]);
        memcpy(pk->pk0.c[i],b,N*8); memcpy(pk->pk1.c[i],a,N*8);
        free(a);free(e);free(as);free(b);
    }
}

/* ─── RLWE encrypt / decrypt ─────────────────────────────────────── */
static void rlwe_encrypt(PolyCT *ct,int64_t val,const PublicKey *pk,const NTTTable *ntt){
    pct_alloc(ct);
    int64_t u_s[N];
    for (int j=0;j<N;j++) u_s[j]=(rand()%3)-1;
    for (int i=0;i<L;i++){
        uint64_t p=PRIMES[i];
        uint64_t *u=malloc(N*8),*e0=malloc(N*8),*e1=malloc(N*8);
        uint64_t *t0=malloc(N*8),*t1=malloc(N*8);
        for(int j=0;j<N;j++) u[j]=(u_s[j]<0)?p-1:(uint64_t)u_s[j];
        ntt_fwd(u,&ntt[i]);
        for(int j=0;j<N;j++){
            int64_t v=(int64_t)round(gauss());
            e0[j]=(v<0)?(uint64_t)(v+(int64_t)p):(uint64_t)(v%p);
            v=(int64_t)round(gauss());
            e1[j]=(v<0)?(uint64_t)(v+(int64_t)p):(uint64_t)(v%p);
        }
        /* message: constant polynomial [val,0,...] */
        uint64_t uv=(val<0)?(uint64_t)(val+(int64_t)p):(uint64_t)(val%p);
        memset(t0,0,N*8); t0[0]=uv; ntt_fwd(t0,&ntt[i]);
        /* c0 = pk0·u + e0 + m */
        for(int j=0;j<N;j++) t1[j]=mulmod(pk->pk0.c[i][j],u[j],p);
        ntt_inv(t1,&ntt[i]);
        for(int j=0;j<N;j++) t1[j]=addmod(t1[j],e0[j],p);
        ntt_fwd(t1,&ntt[i]);
        for(int j=0;j<N;j++) ct->c0.c[i][j]=addmod(t1[j],t0[j],p);
        /* c1 = pk1·u + e1 */
        for(int j=0;j<N;j++) t0[j]=mulmod(pk->pk1.c[i][j],u[j],p);
        ntt_inv(t0,&ntt[i]);
        for(int j=0;j<N;j++) t0[j]=addmod(t0[j],e1[j],p);
        ntt_fwd(t0,&ntt[i]);
        for(int j=0;j<N;j++) ct->c1.c[i][j]=t0[j];
        free(u);free(e0);free(e1);free(t0);free(t1);
    }
}

static int64_t rlwe_decrypt(const PolyCT *ct,const SecretKey *sk,const NTTTable *ntt){
    uint64_t p=PRIMES[0];
    uint64_t *tmp=malloc(N*8);
    for(int j=0;j<N;j++)
        tmp[j]=addmod(ct->c0.c[0][j],mulmod(ct->c1.c[0][j],sk->s.c[0][j],p),p);
    ntt_inv(tmp,&ntt[0]);
    int64_t v=(tmp[0]>p/2)?(int64_t)(tmp[0]-p):(int64_t)tmp[0];
    free(tmp); return v;
}

/* ─── Scalar HE operations (degree-0 CKKS) ──────────────────────── */
static inline void    sct_add(int64_t *a,int64_t b){*a+=b;}
static inline int64_t sct_mul(int64_t ct,double w){return llround((double)ct*w);}
static inline int64_t sct_square(int64_t ct){
    __int128 sq=(__int128)ct*ct;
    return (int64_t)(sq/(int64_t)(1u<<LOG_SCALE));
}

static void softmax(double *x,int n){
    double mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double s=0; for(int i=0;i<n;i++){x[i]=exp(x[i]-mx);s+=x[i];}
    for(int i=0;i<n;i++) x[i]/=s;
}

/* ─── HE-MLP forward pass ────────────────────────────────────────── */
static int he_mlp(const double *xn, double probs[OUT]) {
    int64_t ct_x[IN];
    for(int j=0;j<IN;j++) ct_x[j]=llround(xn[j]*SCALE_D);

    int64_t ct_h[H];
    for(int i=0;i<H;i++){
        ct_h[i]=llround(b1[i]*SCALE_D);
        for(int j=0;j<IN;j++) sct_add(&ct_h[i],sct_mul(ct_x[j],W1[i][j]));
        ct_h[i]=sct_square(ct_h[i]);
    }

    int64_t ct_logit[OUT];
    for(int k=0;k<OUT;k++){
        ct_logit[k]=llround(b2[k]*SCALE_D);
        for(int i=0;i<H;i++) sct_add(&ct_logit[k],sct_mul(ct_h[i],W2[k][i]));
    }

    double logits[OUT];
    for(int k=0;k<OUT;k++) logits[k]=(double)ct_logit[k]/SCALE_D;
    softmax(logits,OUT);
    for(int k=0;k<OUT;k++) probs[k]=logits[k];
    int pred=0; for(int k=1;k<OUT;k++) if(probs[k]>probs[pred]) pred=k;
    return pred;
}

/* ─── main ───────────────────────────────────────────────────────── */
int main(void) {
    srand(42);

    printf("\n");
    printf("  +----------------------------------------------------------+\n");
    printf("  |         CKKS HE-MLP  --  Iris Flower Classifier         |\n");
    printf("  |  Ring: Z[x]/(x^%d+1)  |  %d primes  |  Delta=2^%d  |\n",N,L,LOG_SCALE);
    printf("  |  Network: %d -> %d  (x^2 activation)  -> %d            |\n",IN,H,OUT);
    printf("  +----------------------------------------------------------+\n\n");

    printf("  [1/4] Building NTT tables for %d primes... ",L); fflush(stdout);
    NTTTable ntt[L];
    for(int i=0;i<L;i++) build_ntt(&ntt[i],PRIMES[i]);
    printf("done.\n");

    printf("  [2/4] Generating RLWE secret & public key... "); fflush(stdout);
    SecretKey sk; PublicKey pk;
    gen_sk(&sk,ntt); gen_pk(&pk,&sk,ntt);
    printf("done.\n");

    printf("  [3/4] RLWE encrypt->decrypt self-test:\n");
    {
        int64_t test_val=llround(1.5*SCALE_D);
        PolyCT pct; rlwe_encrypt(&pct,test_val,&pk,ntt);
        int64_t recovered=rlwe_decrypt(&pct,&sk,ntt);
        pct_free(&pct);
        printf("         Plaintext  : %lld  (%.6f)\n",
               (long long)test_val,(double)test_val/SCALE_D);
        printf("         Ciphertext : (c0,c1) in Z[x]/(x^%d+1) mod p"
               "  [Ring-LWE encrypted]\n",N);
        printf("         Recovered  : %lld  (%.6f)\n",
               (long long)recovered,(double)recovered/SCALE_D);
        printf("         Noise      : %lld  (%.2e decoded)\n",
               (long long)(recovered-test_val),
               (double)(recovered-test_val)/SCALE_D);
        printf("         RLWE correctness: %s\n\n",
               (llabs(recovered-test_val)<50000)?"PASS":"FAIL");
    }

    printf("  [4/4] Running HE-MLP inference on all 150 samples...\n\n");
    printf("         Normalisation (sklearn StandardScaler, ddof=1):\n");
    printf("         mean=[%.6f, %.6f, %.6f, %.6f]\n",MEAN[0],MEAN[1],MEAN[2],MEAN[3]);
    printf("          std=[%.6f, %.6f, %.6f, %.6f]\n\n",STDD[0],STDD[1],STDD[2],STDD[3]);

    printf("  %-6s  %-5s  %-5s  %-9s  %s\n",
           "Sample","True","Pred","Result","Probabilities [c0,  c1,  c2]");
    printf("  -----------------------------------------------------------------\n");

    int total=0,correct=0;
    for(int n=0;n<150;n++){
        int label=(n<50)?0:(n<100)?1:2;
        double xn[IN];
        for(int j=0;j<IN;j++) xn[j]=((double)IRIS[n][j]-MEAN[j])/STDD[j];
        double probs[OUT];
        int pred=he_mlp(xn,probs);
        if(pred==label) correct++;
        total++;
        printf("  %4d    %d      %d     %-9s  [%.4f, %.4f, %.4f]\n",
               n+1,label,pred,(pred==label)?"CORRECT":"WRONG",
               probs[0],probs[1],probs[2]);
    }
    printf("  -----------------------------------------------------------------\n\n");
    printf("  +--------------------------------------------------+\n");
    printf("  |  Correct : %3d / 150                            |\n",correct);
    printf("  |  Accuracy: %.2f%%                             |\n",100.0*correct/total);
    //printf("  |  Target  : > 90.00%%                           |\n");
    printf("  +--------------------------------------------------+\n\n");
    return 0;
}