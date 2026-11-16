(set-info :smt-lib-version 2.6)
(set-info :source |The authors Robert Vajda and Zoltan Kovacs released this problem in the paper that can be found in "http://ceur-ws.org/Vol-2752/paper15.pdf". A short description of the problem can be found down below.

Triangle-Bottema5.3:
 Comparison of Expressions Related to Triangle Circumradius via realgeom (Bottema 5.3):Let A, B, C be arbitrary points. Let t1 be the polygon A, B, C. Let c be the segment A, B. Let a be the segment B, C. Let b be the segment C, A. Let f be the perpendicular bisector of c. Let g be the perpendicular bisector of b. Let O be the intersection of f and g. Let R be the segment A, O. Compare a + b + c and segment A, O.

This problem was added to SMT-LIB by Tereso del Rio and Matthew England.| )
(set-info :category "industrial")
(set-info :license "https://creativecommons.org/licenses/by/4.0/")
(set-info :status sat)
(set-logic QF_NRA)
(declare-fun m () Real)
(declare-fun v16 () Real)
(declare-fun v17 () Real)
(declare-fun v18 () Real)
(declare-fun v19 () Real)
(declare-fun v20 () Real)
(declare-fun v21 () Real)
(declare-fun v5 () Real)
(declare-fun v6 () Real)
(declare-fun w1 () Real)
(assert (and (< 0 m) (< 0 v17) (< 0 v19) (< 0 v18) (< 0 v20) (= (+ (* (* v16 v6) 2) (* (* v5 v5) (- 1) )(* (* v6 v6) (- 1) )v5) 0) (= (+ (* (* v18 v18) (- 1) )(* v5 v5) (* v6 v6) (* v5 (- 2) )1) 0) (= (+ (* (* v19 v19) (- 1) )(* v5 v5) (* v6 v6)) 0) (= (+ (* v21 v6) (- 1) )0) (= (+ (* (* v16 v16) 4) (* (* v17 v17) (- 4) )1) 0) (= (+ v18 v19 (* w1 (- 1) )1) 0) (= (+ (* (* m v17) (- 1) )w1) 0) (= (+ (* v20 (- 1) )1) 0)))
(check-sat)
(exit)
