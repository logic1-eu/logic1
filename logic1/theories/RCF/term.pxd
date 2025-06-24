# Singular rings

from cython.cimports.gmpy2 import mpq_t, mpz_ptr, mpz_t

cdef extern from "singular/Singular/libsingular.h":

    # initialization
    int siInit(char *)

    # memory allocation
    void *omAlloc0(size_t size)

    char *omStrDup(char *)

    # rational numbers
    ctypedef struct number "snumber":
        mpz_t z
        mpz_t n
        int s

    # ideals
    ctypedef struct ideal "sip_sideal"

    # polynomial procs
    ctypedef struct p_Procs_s "p_Procs_s"
    
    ctypedef struct n_Procs_s
        
    # ring
    cdef enum rRingOrder_t:
        ringorder_no
        ringorder_a
        ringorder_a64 # for int64 weights
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

    ctypedef struct ring "ip_sring":
        int  *order  # array of orderings
        int  *block0 # starting pos
        int  *block1 # ending pos
        int  **wvhdl # weight vectors
        int  OrdSgn  # 1 for polynomial rings
        int  ShortOut # control printing
        int  CanShortOut # control printing capabilities
        number *minpoly # minpoly for base extension field
        char **names # variable names
        p_Procs_s *p_Procs #polxnomial procs
        ideal *qideal #quotient ideal

        short N # number of variables

        int pCompIndex # index of components
        unsigned long bitmask # mask for getting single exponents

        n_Procs_s* cf  # coefficient field/ring
        int ref

        long (*pLDeg)(poly *p, int *l, ring *r)
        long (*pLDegOrig)(poly *p, int *l, ring *r)
        long (*pFDeg)(poly *p, ring *r)
        long (*pFDegOrig)(poly *p, ring *r)

    void rChangeCurrRing(ring *r)

    ring *rDefault(int ch,
                   int nvars,
                   char **names,
                   int ord_size,
                   rRingOrder_t *ord,
                   int *block0,
                   int *block1,
                   int **wvhdl)

    # polynomials
    ctypedef struct poly "spolyrec"

    # return constant polynomial from int
    poly *p_ISet(int i, ring *r)

    # return constant polynomial from number
    poly *p_NSet(number *n,ring *r)

    int p_SetExp(poly *p, int v, int e, ring *r)

    unsigned long p_GetMaxExp(poly *p, ring *r)
    
    void p_Setm(poly *p, ring *r)

    poly *p_Add_q(poly *p, poly *q, ring *r)     # return p+q, destroys p and q

    poly *pp_Mult_qq(poly *p, poly *q, ring *r)  # return p*q, does neither destroy p nor q

    poly *p_Copy(poly *p, ring *r)


cdef extern from "singular/polys/monomials/ring.h":
    void rPrint "rWrite"(ring* r)

    void pDebugPrint "p_DebugPrint" (poly *p, ring *r)


cdef extern from "singular/polys/monomials/p_polys.h":
    void p_Write(poly *p, ring *lmRing, ring *tailRing)


cdef extern from *:  # hack to get at cython macro
    int unlikely(int)


cdef extern from "singular/coeffs/longrat.h":

    # rational number from numerator and denominator
    number *nlInit2gmp(mpz_t n, mpz_t d,const n_Procs_s* cf)

cdef extern from "gmp.h":

    mpz_ptr mpq_numref (const mpq_t op)

    mpz_ptr mpq_denref (const mpq_t op)
