from cython.cimports.gmpy2 import mpq_t, mpz_ptr, mpz_t

cdef extern from "singular/Singular/libsingular.h":
    
    ctypedef struct ideal "sip_sideal"                          # ideals

    ctypedef struct n_Procs_s:                                  # number procs
        number* cfDiv(number *, number *, const n_Procs_s* r)
        number* cfAdd(number *, number *, const n_Procs_s* r)   # algebraic number addition
        number* cfSub(number *, number *, const n_Procs_s* r)
        number* cfMult(number *, number *, const n_Procs_s* r)  # algebraic number multiplication
        number* (*cfInit)(int i, const n_Procs_s* r ) # algebraic number from int
        number* (*cfInitMPZ)(mpz_t i, const n_Procs_s* r)
        number* (*cfParameter)(int i, const n_Procs_s* r)
        int (*cfParDeg)(number* n, const n_Procs_s* r)
        int (*cfSize)(number* n, const n_Procs_s* r)
        int (*cfInt)(number* n, const n_Procs_s* r)
        int (*cdDivComp)(number* a,number* b, const n_Procs_s* r)
        number* (*cfGetUnit)(number* a, const n_Procs_s* r)
        number* (*cfExtGcd)(number* a, number* b, number* *s, number* *t , const n_Procs_s* r)
        void (*cfDelete)(number **, const n_Procs_s*)
        number* (*cfInpNeg)(number* a,  const n_Procs_s* r)
        number* (*cfInvers)(number* a,  const n_Procs_s* r)
        number* (*cfCopy)(number* a,  const n_Procs_s* r)       # deep copy of algebraic number
        number* (*cfRePart)(number* a, const n_Procs_s* cf)
        number* (*cfImPart)(number* a, const n_Procs_s* cf)
        void (*cfWrite)(number* a, const n_Procs_s* r)
        void (*cfNormalize)(number* a,  const n_Procs_s* r)
        bint (*cfDivBy)(number* a, number* b, const n_Procs_s* r)
        bint (*cfGreater)(number* a, number* b, const n_Procs_s* )
        bint (*cfEqual)(number* a,number* b, const n_Procs_s* )
        bint (*cfIsZero)(number* a, const n_Procs_s* )          # algebraic number comparison with zero
        bint (*cfIsOne)(number* a, const n_Procs_s* )           # algebraic number comparison with one
        bint (*cfIsMOne)(number* a, const n_Procs_s* )
        bint (*cfGreaterZero)(number* a, const n_Procs_s* )
        void (*cfPower)(number* a, int i, number* * result,
              const n_Procs_s* r)                               # algebraic number power
        ring *extRing
        int ch
        mpz_ptr modBase
        unsigned long modExponent
        int type

    ctypedef struct number "snumber":                           # rational numbers
        mpz_t z                                                 # numerator
        mpz_t n                                                 # denominator
        int s
  
    ctypedef struct p_Procs_s "p_Procs_s"                       # polynomial procs
    
    ctypedef struct poly "spolyrec"                             # polynomials

    ctypedef struct ring "ip_sring":                            # rings
        int  *order            # array of orderings
        int  *block0           # starting pos
        int  *block1           # ending pos
        int  **wvhdl           # weight vectors
        int  OrdSgn            # 1 for polynomial rings
        int  ShortOut          # control printing
        int  CanShortOut       # control printing capabilities
        number *minpoly        # minpoly for base extension field
        char **names           # variable names
        p_Procs_s *p_Procs     # polxnomial procs
        ideal *qideal          # quotient
        short N                # number of variables
        int pCompIndex         # index of components
        unsigned long bitmask  # mask for getting single expo
        n_Procs_s* cf          # coefficient field/ring
        int ref

        # return total degree of p
        long (*pLDeg)(poly *p, int *l, ring *r)
        long (*pLDegOrig)(poly *p, int *l, ring *r)
        long (*pFDeg)(poly *p, ring *r)
        long (*pFDegOrig)(poly *p, ring *r)
    
    cdef enum rRingOrder_t:                                     # available ring orders
        ringorder_no
        ringorder_a
        ringorder_a64  # for int64 weights
        ringorder_c
        ringorder_C
        ringorder_M
        ringorder_S
        ringorder_s
        ringorder_lp
        ringorder_dp
        ringorder_ip
        ringorder_Dp
        ringorder_wp
        ringorder_Wp
        ringorder_ls
        ringorder_ds
        ringorder_Ds
        ringorder_ws
        ringorder_Ws
        ringorder_L

    cdef long SR_INT                                            # integer conversion constant

    void n_Delete(number **n, n_Procs_s *cf)                    # general number destructor
    void *omAlloc0(size_t size)                                 # memory allocation
    char *omStrDup(char *)
    unsigned long p_GetMaxExp(poly *p, ring *r)                 # get the maximal exponent in p
    poly *p_Add_q(poly *p, poly *q, ring *r)                    # return p+q, destroys p and q
    poly *p_Copy(poly *p, ring *r)                              # deep copy p
    long p_Deg(poly *p, ring *r)                                # total degree of the leading monomial
    int p_EqualPolys(poly *p1, poly *p2, const ring *r)
    number *p_GetCoeff(poly *p, ring *r)                        # get the coefficient of the current list element p in r
    int p_GetExp(poly *p, int v, ring *r)                       # get the exponent at index v of the monomial p in r, v starts at 1
    poly *p_Head(poly *p, ring *r)                              # return new copy of lm(p), coefficient copied, next=NULL, p may be NULL
    poly *p_Init(ring *r)                                       # return new empty monomial
    int p_IsConstant(poly *, ring *)                            # TRUE if poly is constant
    poly *p_ISet(int i, ring *r)                                # return constant polynomial from int
    poly *p_Neg(poly *p, ring *r)                               # return -p, p is destroyed
    poly *p_NSet(number *n,ring *r)                             # return constant polynomial from number
    int p_SetCoeff(poly *p, number *n, ring *r)                 # set the coefficient n for the current list element p in r
    int p_SetExp(poly *p, int v, int e, ring *r)                # set the exponent e at index v for the monomial p, v starts at 1
    void p_Setm(poly *p, ring *r)                               # if SetExp is called on p, p_Setm needs to be called afterwards to finalize the change.
    char *p_String(poly *p, ring *r, ring *r)                   # return string representation of p
    poly *p_Sub(poly *p1, poly *p2, const ring *r)              # subtract p2 from p1, p1 and p2 are destroyed
    poly *pp_Mult_nn(poly *p, number *n, ring *r)               # return p*n, p is const (i.e. copied)
    poly *pp_Mult_qq(poly *p, poly *q, ring *r)                 # return p*q, does neither destroy p nor q
    poly *pNext(poly *p)                                        # iterate through the monomials of p
    void rChangeCurrRing(ring *r)
    ring *rDefault(int ch, int nvars, char **names,
                   int ord_size, rRingOrder_t *ord,
                   int *block0, int *block1, int **wvhdl)       # construct ring with characteristic, number of vars and names
    long SR_TO_INT(number *)                                    # number to integer handle
    long SR_HDL(number *)                                       # available ring orders
    int siInit(char *)                                          # initialization

cdef extern from "singular/coeffs/longrat.h":
    number *nlGetNumerator(number *n, const n_Procs_s *cf)      # available ring orders
    number *nlGetDenom(number *n, const n_Procs_s *cf)          # get denominator
    number *nlInit2gmp(mpz_t n, mpz_t d,const n_Procs_s *cf)    # rational number from numerator and denominator
    void nlDelete(number **n, const n_Procs_s *cf)              # delete rational number

cdef extern from "singular/polys/monomials/p_polys.h":
    void p_Write(poly *p, ring *lmRing, ring *tailRing)

cdef extern from "singular/polys/monomials/ring.h":
    void rPrint "rWrite"(ring *r)
    char *rString(ring *r)
    void pDebugPrint "p_DebugPrint" (poly *p, ring *r)

# cdef extern from "singular/polys/sbuckets.h":
#     
#     ctypedef struct sBucket:                                    # sBucket is actually a class
#         pass
#     
#     sBucket *sBucketCreate(ring *r)                             # create an sBucket

cdef extern from "gmp.h":
    mpz_ptr mpq_numref (const mpq_t op)
    mpz_ptr mpq_denref (const mpq_t op)
    void mpz_set (mpz_t rop, const mpz_t op)
    void mpz_set_si (mpz_t rop, signed long int op)
    void mpq_set_den (mpq_t rational, const mpz_t denominator)
    void mpq_set_num (mpq_t rational, const mpz_t numerator)

cdef extern from *:  # hack to get at cython macro
    int unlikely(int)