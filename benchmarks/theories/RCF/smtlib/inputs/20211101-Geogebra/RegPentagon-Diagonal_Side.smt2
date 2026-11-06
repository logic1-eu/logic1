(set-info :smt-lib-version 2.6)
(set-info :source |The authors Robert Vajda and Zoltan Kovacs released this problem in the paper that can be found in "http://ceur-ws.org/Vol-2752/paper15.pdf". A short description of the problem can be found down below.

RegPentagon-Diagonal_Side:
 Comparison of Diagonal and Side (regular pentagon):Let A, B be arbitrary points. Let poly1 be the regular 5-gon with vertices A, B, C, D, E. Let f be the segment A, B. Let k be the segment E, C. Compare segment E, C and segment A, B.

This problem was added to SMT-LIB by Tereso del Rio and Matthew England.| )
(set-info :category "industrial")
(set-info :license "https://creativecommons.org/licenses/by/4.0/")
(set-info :status sat)
(set-logic QF_NRA)
(declare-fun m () Real)
(declare-fun v10 () Real)
(declare-fun v11 () Real)
(declare-fun v12 () Real)
(declare-fun v14 () Real)
(declare-fun v15 () Real)
(declare-fun v7 () Real)
(declare-fun v8 () Real)
(declare-fun v9 () Real)
(assert (and (< 0 m) (< 0 v14) (< 0 v15) (= (+ (* v7 v7) (* (* v8 v8) (- 1) )(* v7 (- 1) )(* v9 (- 1) )1) 0) (= (- (* (* v7 v8) 2) v10 v8) 0) (= (+ (* (* v10 v8) (- 1) )(* (* v7 v7) (- 1) )(* v7 v9) (* v8 v8) (* v11 (- 1) )v7) 0) (= (+ (* v10 v7) (* (* v7 v8) (- 2) )(* v8 v9) (* v12 (- 1) )v8) 0) (= (+ (* v11 v11) (* (* v11 v7) (- 2) )(* v12 v12) (* (* v8 v12) (- 2) )(* (* v14 v14) (- 1) )(* v7 v7) (* v8 v8)) 0) (= (+ (* v7 v7) (* v8 v8) (* v7 (- 2))) 0) (= (+ (* (* v7 v7) 4) (* v7 (- 6) )1) 0) (= (+ (* m (- 1) )v14) 0) (= (+ (* v15 (- 1) )1) 0)))
(check-sat)
(exit)
