(set-info :smt-lib-version 2.6)
(set-logic QF_NRA)
(set-info :author |Thomas Sturm, CNRS France, http://science.thomas-sturm.de|)
(set-info :source |
Transmitted by: Thomas Sturm
Generated on: 20161105
Received on: 20161105
Generator: RedLog, http://www.redlog.eu/
Application: qualitative analysis of systems of ODE in physics, chemistry, and the life sciences
Publication: Algebraic Biology 2008, doi:10.1007/978-3-540-85101-1_15
Additional information:
All problems have the following form: a certain polynomial has a zero where all variables are positive.  Interval solutions for satisfiable problems is a valuable information.
|)
(set-info :series |MBO - Methylene Blue Oscillator System|)
(set-info :instance |E4E11|)
(set-info :category "industrial")
(set-info :status sat)
(declare-const j2 Real)
(declare-const h6 Real)
(declare-const h5 Real)
(declare-const h4 Real)
(declare-const h3 Real)
(declare-const h2 Real)
(declare-const h1 Real)
(assert (and (> h1 0) (> h2 0) (> h3 0) (> h4 0) (> h5 0) (> h6 0) (> j2 0) (= 
(+ (* 36 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2)) (* 156 (* h1
 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2)) (* 232 (* h1 h1 h1 h1) (* h2 
h2 h2) (* h3 h3) h5 (* j2 j2)) (* 144 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 
j2) (* 32 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5) (* 6 (* h1 h1 h1 h1) (* h2 
h2 h2) h3 (* h5 h5) (* j2 j2 j2 j2)) (* 28 (* h1 h1 h1 h1) (* h2 h2 h2) h3 (* h5
 h5) (* j2 j2 j2)) (* 46 (* h1 h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* j2 j2)) 
(* 32 (* h1 h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) j2) (* 8 (* h1 h1 h1 h1) (* h2 
h2 h2) h3 (* h5 h5)) (* (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2
)) (* 5 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2)) (* 9 (* h1 h1 h1
 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2)) (* 7 (* h1 h1 h1 h1) (* h2 h2 h2) (* 
h5 h5) h6 j2) (* 2 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6) (* 180 (* h1 h1 h1
 h1) (* h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 888 (* h1 h1 h1 h1) (* h2 h2)
 (* h3 h3 h3) h5 (* j2 j2 j2)) (* 1520 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5
 (* j2 j2)) (* 1056 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 j2) (* 256 (* h1 
h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5) (* 78 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) 
h4 h5 (* j2 j2 j2 j2)) (* 334 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 j2
 j2)) (* 488 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 j2)) (* 296 (* h1 
h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 j2) (* 64 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3
) h4 h5) (* 54 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) 
(* 234 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2)) (* 360 (* h1 
h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2)) (* 248 (* h1 h1 h1 h1) (* h2 
h2) (* h3 h3) (* h5 h5) j2) (* 64 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5))
 (* 54 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 222 (* h1 
h1 h1 h1) (* h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2)) (* 304 (* h1 h1 h1 h1) (* h2 
h2) (* h3 h3) h5 h6 (* j2 j2)) (* 168 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h5 h6 
j2) (* 32 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h5 h6) (* 15 (* h1 h1 h1 h1) (* h2
 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2)) (* 62 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* 
h5 h5) (* j2 j2 j2)) (* 85 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2)) 
(* 46 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) j2) (* 8 (* h1 h1 h1 h1) (* h2 
h2) h3 h4 (* h5 h5)) (- (* 2 (* h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2 
j2))) (- (* 8 (* h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2))) (- (* 10 (* 
h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2))) (- (* 4 (* h1 h1 h1 h1) (* h2 h2)
 h3 h4 h5 h6 j2)) (* 6 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2)
) (* 28 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2)) (* 46 (* h1 h1 
h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* j2 j2)) (* 32 (* h1 h1 h1 h1) (* h2 h2) h3 
(* h5 h5 h5) j2) (* 8 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5)) (* 7 (* h1 h1 
h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 28 (* h1 h1 h1 h1) (* h2 h2
) h3 (* h5 h5) h6 (* j2 j2 j2)) (* 39 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 
(* j2 j2)) (* 26 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 j2) (* 8 (* h1 h1 h1 
h1) (* h2 h2) h3 (* h5 h5) h6) (- (* 2 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6)
 (* j2 j2 j2 j2))) (- (* 8 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2
))) (- (* 10 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2))) (- (* 4 (* h1
 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) j2)) (* 2 (* h1 h1 h1 h1) (* h2 h2) h4 (* 
h5 h5) h6 (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 
j2 j2)) (* 14 (* h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2)) (* 9 (* h1 h1
 h1 h1) (* h2 h2) h4 (* h5 h5) h6 j2) (* 2 (* h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5
) h6) (* (* h1 h1 h1 h1) (* h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 5 (* h1 
h1 h1 h1) (* h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 9 (* h1 h1 h1 h1) (* h2 h2)
 (* h5 h5 h5) h6 (* j2 j2)) (* 7 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5 h5) h6 j2) 
(* 2 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5 h5) h6) (* (* h1 h1 h1 h1) (* h2 h2) (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5) (* h6
 h6) (* j2 j2 j2)) (* 5 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6) (* j2 j2))
 (* 2 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6) j2) (* 210 (* h1 h1 h1 h1) 
h2 (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 1016 (* h1 h1 h1 h1) h2 (* h3 h3 h3) 
h4 h5 (* j2 j2 j2)) (* 1688 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 (* j2 j2)) (* 
1120 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 j2) (* 256 (* h1 h1 h1 h1) h2 (* h3 
h3 h3) h4 h5) (* 120 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) 
(* 632 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 1184 (* h1 h1 
h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 j2)) (* 928 (* h1 h1 h1 h1) h2 (* h3 h3 
h3) (* h5 h5) j2) (* 256 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5)) (* 90 (* h1 
h1 h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 384 (* h1 h1 h1 h1) h2 (* h3
 h3 h3) h5 h6 (* j2 j2 j2)) (* 504 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 
j2)) (* 192 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h5 h6 j2) (* 54 (* h1 h1 h1 h1) h2 
(* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 226 (* h1 h1 h1 h1) h2 (* h3 h3) (* 
h4 h4) h5 (* j2 j2 j2)) (* 316 (* h1 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 
j2)) (* 176 (* h1 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 j2) (* 32 (* h1 h1 h1 h1) 
h2 (* h3 h3) (* h4 h4) h5) (* 87 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* j2
 j2 j2 j2)) (* 367 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 
530 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* j2 j2)) (* 320 (* h1 h1 h1 h1) 
h2 (* h3 h3) h4 (* h5 h5) j2) (* 64 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5)) 
(* 78 (* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 326 (* h1 h1 h1 
h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 464 (* h1 h1 h1 h1) h2 (* h3 h3) h4 
h5 h6 (* j2 j2)) (* 280 (* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 j2) (* 64 (* h1 h1
 h1 h1) h2 (* h3 h3) h4 h5 h6) (* 24 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) 
(* j2 j2 j2 j2)) (* 112 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 j2 j2)) 
(* 184 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 j2)) (* 128 (* h1 h1 h1 
h1) h2 (* h3 h3) (* h5 h5 h5) j2) (* 32 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5
)) (* 50 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 210 (* h1
 h1 h1 h1) h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 312 (* h1 h1 h1 h1) h2 (* 
h3 h3) (* h5 h5) h6 (* j2 j2)) (* 216 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5) h6 
j2) (* 64 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5) h6) (* 24 (* h1 h1 h1 h1) h2 
(* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 100 (* h1 h1 h1 h1) h2 (* h3 h3) h5 
(* h6 h6) (* j2 j2 j2)) (* 148 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) (* j2 
j2)) (* 104 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) j2) (* 32 (* h1 h1 h1 h1) 
h2 (* h3 h3) h5 (* h6 h6)) (* 12 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2
 j2 j2 j2)) (* 43 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 45 
(* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2)) (* 14 (* h1 h1 h1 h1) h2 h3
 (* h4 h4) (* h5 h5) j2) (- (* 2 (* h1 h1 h1 h1) h2 h3 (* h4 h4) h5 h6 (* j2 j2 
j2 j2))) (- (* 8 (* h1 h1 h1 h1) h2 h3 (* h4 h4) h5 h6 (* j2 j2 j2))) (- (* 10 
(* h1 h1 h1 h1) h2 h3 (* h4 h4) h5 h6 (* j2 j2))) (- (* 4 (* h1 h1 h1 h1) h2 h3 
(* h4 h4) h5 h6 j2)) (* 9 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2))
 (* 34 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) (* j2 j2 j2)) (* 39 (* h1 h1 h1 h1)
 h2 h3 h4 (* h5 h5 h5) (* j2 j2)) (* 14 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) j2
) (* 21 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 87 (* h1 h1 h1
 h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2)) (* 122 (* h1 h1 h1 h1) h2 h3 h4 (* h5 
h5) h6 (* j2 j2)) (* 72 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5) h6 j2) (* 16 (* h1 h1
 h1 h1) h2 h3 h4 (* h5 h5) h6) (- (* 4 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* 
j2 j2 j2 j2))) (- (* 16 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2 j2))) (- 
(* 20 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2))) (- (* 8 (* h1 h1 h1 h1) 
h2 h3 h4 h5 (* h6 h6) j2)) (* 8 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 
j2 j2)) (* 40 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 72 (* h1 h1
 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2)) (* 56 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5
) h6 j2) (* 16 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6) (* 7 (* h1 h1 h1 h1) h2 h3
 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* 
h6 h6) (* j2 j2 j2)) (* 51 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6) (* j2 j2)) 
(* 34 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6) j2) (* 8 (* h1 h1 h1 h1) h2 h3 
(* h5 h5) (* h6 h6)) (- (* 2 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2 
j2))) (- (* 8 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2))) (- (* 10 (* 
h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2))) (- (* 4 (* h1 h1 h1 h1) h2 h3 h5 
(* h6 h6 h6) j2)) (* (* h1 h1 h1 h1) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) 
(* 4 (* h1 h1 h1 h1) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 5 (* h1 h1 h1 h1
) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 2 (* h1 h1 h1 h1) h2 (* h4 h4) (* h5 
h5) h6 j2) (* (* h1 h1 h1 h1) h2 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 4 (* h1 
h1 h1 h1) h2 h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 5 (* h1 h1 h1 h1) h2 h4 (* h5 
h5 h5) h6 (* j2 j2)) (* 2 (* h1 h1 h1 h1) h2 h4 (* h5 h5 h5) h6 j2) (* 2 (* h1 
h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h2 h4 
(* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 14 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6
) (* j2 j2)) (* 9 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) j2) (* 2 (* h1 h1 h1
 h1) h2 h4 (* h5 h5) (* h6 h6)) (* (* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 9 
(* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 7 (* h1 h1 h1 h1) h2 (* 
h5 h5 h5) (* h6 h6) j2) (* 2 (* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6)) (* (* h1
 h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) h2 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h2 (* h5 h5) (* h6 h6 
h6) (* j2 j2)) (* 7 (* h1 h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6) j2) (* 2 (* h1 h1 
h1 h1) h2 (* h5 h5) (* h6 h6 h6)) (* 60 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4) 
h5 (* j2 j2 j2 j2)) (* 276 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2
)) (* 408 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2)) (* 192 (* h1 h1 
h1 h1) (* h3 h3 h3) (* h4 h4) h5 j2) (* 60 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5
 h5) (* j2 j2 j2 j2)) (* 276 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 
j2)) (* 408 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 192 (* h1 h1
 h1 h1) (* h3 h3 h3) h4 (* h5 h5) j2) (* 140 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 
h6 (* j2 j2 j2 j2)) (* 724 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) 
(* 1320 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 992 (* h1 h1 h1 h1) 
(* h3 h3 h3) h4 h5 h6 j2) (* 256 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6) (* 80 
(* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 448 (* h1 h1 h1 h1
) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 912 (* h1 h1 h1 h1) (* h3 h3 h3) 
(* h5 h5) h6 (* j2 j2)) (* 800 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 j2) (* 
256 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6) (* 80 (* h1 h1 h1 h1) (* h3 h3 h3
) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 448 (* h1 h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6)
 (* j2 j2 j2)) (* 912 (* h1 h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 
800 (* h1 h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6) j2) (* 256 (* h1 h1 h1 h1) (* h3 
h3 h3) h5 (* h6 h6)) (* 12 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2
 j2)) (* 48 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 60 (* h1 
h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 24 (* h1 h1 h1 h1) (* h3 h3) 
(* h4 h4 h4) h5 j2) (* 30 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2
 j2 j2)) (* 114 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 
132 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2)) (* 48 (* h1 h1 h1 
h1) (* h3 h3) (* h4 h4) (* h5 h5) j2) (* 40 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) 
h5 h6 (* j2 j2 j2 j2)) (* 176 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2
 j2)) (* 264 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 160 (* h1 
h1 h1 h1) (* h3 h3) (* h4 h4) h5 h6 j2) (* 32 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4
) h5 h6) (* 12 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 48 
(* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 60 (* h1 h1 h1 h1) 
(* h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 24 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5
 h5) j2) (* 67 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 297
 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 454 (* h1 h1 h1 h1) 
(* h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 288 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 
h5) h6 j2) (* 64 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6) (* 44 (* h1 h1 h1 h1
) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 208 (* h1 h1 h1 h1) (* h3 h3) h4
 h5 (* h6 h6) (* j2 j2 j2)) (* 348 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6) (* 
j2 j2)) (* 248 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6) j2) (* 64 (* h1 h1 h1 
h1) (* h3 h3) h4 h5 (* h6 h6)) (* 16 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2)) (* 80 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) 
(* 144 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 112 (* h1 h1 h1 
h1) (* h3 h3) (* h5 h5 h5) h6 j2) (* 32 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) 
h6) (* 36 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 176 
(* h1 h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 308 (* h1 h1 h1 
h1) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 232 (* h1 h1 h1 h1) (* h3 h3) 
(* h5 h5) (* h6 h6) j2) (* 64 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6)) (* 
16 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 80 (* h1 h1 h1 
h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 144 (* h1 h1 h1 h1) (* h3 h3) h5 
(* h6 h6 h6) (* j2 j2)) (* 112 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) j2) (* 
32 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6)) (* 3 (* h1 h1 h1 h1) h3 (* h4 h4 
h4) (* h5 h5) (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) 
(* j2 j2 j2)) (* 6 (* h1 h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2)) (* 3 (* 
h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h3 
(* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 6 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 
h5) (* j2 j2)) (* 10 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) 
(* 37 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 41 (* h1 h1 h1 
h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 14 (* h1 h1 h1 h1) h3 (* h4 h4) (* 
h5 h5) h6 j2) (* 7 (* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 28 
(* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 35 (* h1 h1 h1 h1) h3 h4 
(* h5 h5 h5) h6 (* j2 j2)) (* 14 (* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 j2) (* 11
 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 48 (* h1 h1 h1 h1
) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 71 (* h1 h1 h1 h1) h3 h4 (* h5 h5) 
(* h6 h6) (* j2 j2)) (* 42 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6) j2) (* 8 
(* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6)) (* 4 (* h1 h1 h1 h1) h3 (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2)) (* 20 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2
 j2 j2)) (* 36 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 28 (* h1 
h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) j2) (* 8 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* 
h6 h6)) (* 4 (* h1 h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 20 
(* h1 h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 36 (* h1 h1 h1 h1) h3
 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 28 (* h1 h1 h1 h1) h3 (* h5 h5) (* h6 h6 
h6) j2) (* 8 (* h1 h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6)) (* 36 (* h1 h1 h1) (* h2
 h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2)) (* 120 (* h1 h1 h1) (* h2 h2 h2 h2) (* 
h3 h3) h5 (* j2 j2 j2)) (* 148 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 
j2)) (* 80 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 j2) (* 16 (* h1 h1 h1) (* 
h2 h2 h2 h2) (* h3 h3) h5) (* 6 (* h1 h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) (* j2 
j2 j2 j2)) (* 22 (* h1 h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) (* j2 j2 j2)) (* 30 
(* h1 h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) (* j2 j2)) (* 18 (* h1 h1 h1) (* h2 h2
 h2 h2) h3 (* h5 h5) j2) (* 4 (* h1 h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5)) (* (* 
h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 
h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2)) (* 6 (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5)
 h6 (* j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 j2) (* (* h1 h1 h1
) (* h2 h2 h2 h2) (* h5 h5) h6) (* 432 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5
 (* j2 j2 j2 j2)) (* 1584 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2)
) (* 2112 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2)) (* 1216 (* h1 h1 
h1) (* h2 h2 h2) (* h3 h3 h3) h5 j2) (* 256 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 
h3) h5) (* 186 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 760
 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2)) (* 1090 (* h1 h1 h1) 
(* h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2)) (* 660 (* h1 h1 h1) (* h2 h2 h2) (* h3 
h3) h4 h5 j2) (* 144 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5) (* 132 (* h1 h1 
h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 496 (* h1 h1 h1) (* h2 
h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2)) (* 692 (* h1 h1 h1) (* h2 h2 h2) (* h3 
h3) (* h5 h5) (* j2 j2)) (* 424 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) j2
) (* 96 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5)) (* 126 (* h1 h1 h1) (* h2
 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 516 (* h1 h1 h1) (* h2 h2 h2) (* h3 
h3) h5 h6 (* j2 j2 j2)) (* 738 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 h6 (* j2 
j2)) (* 444 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 h6 j2) (* 96 (* h1 h1 h1) (* 
h2 h2 h2) (* h3 h3) h5 h6) (* 33 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2
 j2 j2 j2)) (* 137 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2)) (* 
207 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2)) (* 135 (* h1 h1 h1) (* 
h2 h2 h2) h3 h4 (* h5 h5) j2) (* 32 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5)) 
(- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2))) (- (* 6 (* h1 h1
 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2))) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) 
h3 h4 h5 h6 (* j2 j2))) (- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 j2)) (* 12
 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 50 (* h1 h1 h1) 
(* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2)) (* 76 (* h1 h1 h1) (* h2 h2 h2) h3 
(* h5 h5 h5) (* j2 j2)) (* 50 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) j2) (* 
12 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5)) (* 20 (* h1 h1 h1) (* h2 h2 h2) h3
 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 84 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) h6 
(* j2 j2 j2)) (* 132 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) h6 (* j2 j2)) (* 92 
(* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) h6 j2) (* 24 (* h1 h1 h1) (* h2 h2 h2) h3
 (* h5 h5) h6) (- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2 j2)
)) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2))) (- (* 6 (* 
h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 j2))) (- (* 2 (* h1 h1 h1) (* h2 h2
 h2) h3 h5 (* h6 h6) j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 
j2 j2 j2)) (* 18 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 30 
(* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2)) (* 22 (* h1 h1 h1) (* h2 h2
 h2) h4 (* h5 h5) h6 j2) (* 6 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6) (* 2 
(* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1) (* 
h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 15 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5
 h5) h6 (* j2 j2)) (* 11 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 j2) (* 3 (* 
h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6) (* 2 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5)
 (* h6 h6) (* j2 j2 j2 j2)) (* 9 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) 
(* j2 j2 j2)) (* 15 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2)) (* 
11 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) j2) (* 3 (* h1 h1 h1) (* h2 h2 
h2) (* h5 h5) (* h6 h6)) (* 180 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 
j2 j2 j2)) (* 1248 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2)) (* 
2576 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2)) (* 1984 (* h1 h1 h1) 
(* h2 h2) (* h3 h3 h3 h3) h5 j2) (* 512 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) 
h5) (* 936 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 4104 
(* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 6544 (* h1 h1 h1) (* 
h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2)) (* 4352 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3)
 h4 h5 j2) (* 1024 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5) (* 498 (* h1 h1 h1
) (* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 2258 (* h1 h1 h1) (* h2 
h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 3708 (* h1 h1 h1) (* h2 h2) (* h3 h3
 h3) (* h5 h5) (* j2 j2)) (* 2576 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) 
j2) (* 640 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5)) (* 468 (* h1 h1 h1) 
(* h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 2064 (* h1 h1 h1) (* h2 h2) (* 
h3 h3 h3) h5 h6 (* j2 j2 j2)) (* 3328 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 h6 
(* j2 j2)) (* 2208 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 h6 j2) (* 512 (* h1 h1
 h1) (* h2 h2) (* h3 h3 h3) h5 h6) (* 210 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4
 h4) h5 (* j2 j2 j2 j2)) (* 918 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 
(* j2 j2 j2)) (* 1352 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2)) 
(* 820 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 j2) (* 176 (* h1 h1 h1) (* 
h2 h2) (* h3 h3) (* h4 h4) h5) (* 336 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 
h5) (* j2 j2 j2 j2)) (* 1362 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2
 j2 j2)) (* 2050 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2)) (* 
1368 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) j2) (* 336 (* h1 h1 h1) (* h2
 h2) (* h3 h3) h4 (* h5 h5)) (* 316 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 h6 
(* j2 j2 j2 j2)) (* 1404 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2))
 (* 2076 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2)) (* 1260 (* h1 h1 
h1) (* h2 h2) (* h3 h3) h4 h5 h6 j2) (* 272 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 
h5 h6) (* 96 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 
412 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2)) (* 604 (* h1 h1 
h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2)) (* 392 (* h1 h1 h1) (* h2 h2) 
(* h3 h3) (* h5 h5 h5) j2) (* 96 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5)) 
(* 201 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 856 (* 
h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 1307 (* h1 h1 h1) 
(* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2)) (* 880 (* h1 h1 h1) (* h2 h2) (* h3 
h3) (* h5 h5) h6 j2) (* 224 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6) (* 
106 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 486 (* h1 
h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 724 (* h1 h1 h1) (* h2 
h2) (* h3 h3) h5 (* h6 h6) (* j2 j2)) (* 440 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5
 (* h6 h6) j2) (* 96 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6)) (* 42 (* h1 
h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 170 (* h1 h1 h1) (* 
h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 228 (* h1 h1 h1) (* h2 h2) h3 (* 
h4 h4) (* h5 h5) (* j2 j2)) (* 120 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5)
 j2) (* 20 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5)) (- (* 4 (* h1 h1 h1) 
(* h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2))) (- (* 16 (* h1 h1 h1) (* h2 h2) 
h3 (* h4 h4) h5 h6 (* j2 j2 j2))) (- (* 20 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) 
h5 h6 (* j2 j2))) (- (* 8 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) h5 h6 j2)) (* 30 
(* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 127 (* h1 h1 h1) 
(* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2)) (* 198 (* h1 h1 h1) (* h2 h2) h3 h4 
(* h5 h5 h5) (* j2 j2)) (* 133 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) j2) (* 
32 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5)) (* 76 (* h1 h1 h1) (* h2 h2) h3 h4
 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 307 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) h6 
(* j2 j2 j2)) (* 422 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2)) (* 247
 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) h6 j2) (* 56 (* h1 h1 h1) (* h2 h2) h3 
h4 (* h5 h5) h6) (- (* 8 (* h1 h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6) (* j2 j2 j2 
j2))) (- (* 32 (* h1 h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6) (* j2 j2 j2))) (- (* 40
 (* h1 h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6) (* j2 j2))) (- (* 16 (* h1 h1 h1) (* 
h2 h2) h3 h4 h5 (* h6 h6) j2)) (* 6 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) 
(* j2 j2 j2)) (* 16 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2)) (* 14 
(* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) j2) (* 4 (* h1 h1 h1) (* h2 h2) h3 (* 
h5 h5 h5 h5)) (* 30 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 132 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 204 (* h1 h1 
h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2)) (* 138 (* h1 h1 h1) (* h2 h2) h3 (* 
h5 h5 h5) h6 j2) (* 36 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6) (* 28 (* h1 h1
 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 114 (* h1 h1 h1) (* h2
 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 158 (* h1 h1 h1) (* h2 h2) h3 (* h5
 h5) (* h6 h6) (* j2 j2)) (* 96 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) j2
) (* 24 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6)) (- (* 4 (* h1 h1 h1) (* 
h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2))) (- (* 16 (* h1 h1 h1) (* h2 h2) h3 
h5 (* h6 h6 h6) (* j2 j2 j2))) (- (* 20 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6
) (* j2 j2))) (- (* 8 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6) j2)) (* 3 (* h1 
h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 14 (* h1 h1 h1) (* 
h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 22 (* h1 h1 h1) (* h2 h2) (* h4 
h4) (* h5 h5) h6 (* j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 
j2) (* 3 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6) (* 3 (* h1 h1 h1) (* h2 
h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5
 h5) h6 (* j2 j2 j2)) (* 25 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2))
 (* 20 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 j2) (* 6 (* h1 h1 h1) (* h2 h2)
 h4 (* h5 h5 h5) h6) (* 6 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2
 j2 j2)) (* 28 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 44
 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 28 (* h1 h1 h1) (* 
h2 h2) h4 (* h5 h5) (* h6 h6) j2) (* 6 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6
 h6)) (* (* h1 h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 3 (* h1 h1 
h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 3 (* h1 h1 h1) (* h2 h2) (* h5 h5
 h5 h5) h6 j2) (* (* h1 h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6) (* 3 (* h1 h1 h1) 
(* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 23 (* h1 h1 h1) (* h2 h2) (* h5 h5 h5) 
(* h6 h6) (* j2 j2)) (* 16 (* h1 h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6) j2) (* 
4 (* h1 h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6)) (* 3 (* h1 h1 h1) (* h2 h2) (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* 
h6 h6 h6) (* j2 j2 j2)) (* 22 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* 
j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) j2) (* 3 (* h1 h1 h1
) (* h2 h2) (* h5 h5) (* h6 h6 h6)) (* 210 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5
 (* j2 j2 j2 j2)) (* 1436 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) 
(* 2880 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2)) (* 2112 (* h1 h1 h1) h2
 (* h3 h3 h3 h3) h4 h5 j2) (* 512 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5) (* 120 
(* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 872 (* h1 h1 h1) 
h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 1968 (* h1 h1 h1) h2 (* h3 h3 h3 
h3) (* h5 h5) (* j2 j2)) (* 1728 (* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) j2) 
(* 512 (* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5)) (* 90 (* h1 h1 h1) h2 (* h3 h3
 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 564 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* 
j2 j2 j2)) (* 912 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2)) (* 384 (* h1 
h1 h1) h2 (* h3 h3 h3 h3) h5 h6 j2) (* 438 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4
) h5 (* j2 j2 j2 j2)) (* 2104 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 (* j2 j2
 j2)) (* 3544 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 (* j2 j2)) (* 2336 (* h1
 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 j2) (* 512 (* h1 h1 h1) h2 (* h3 h3 h3) (* 
h4 h4) h5) (* 489 (* h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 
2527 (* h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 4740 (* h1 h1 h1
) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 3744 (* h1 h1 h1) h2 (* h3 h3 h3) 
h4 (* h5 h5) j2) (* 1024 (* h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5)) (* 866 (* h1
 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 4126 (* h1 h1 h1) h2 (* h3 
h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 6888 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* 
j2 j2)) (* 4544 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 j2) (* 1024 (* h1 h1 h1) 
h2 (* h3 h3 h3) h4 h5 h6) (* 48 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2 
j2 j2 j2)) (* 368 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2)) (* 896
 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2 j2)) (* 832 (* h1 h1 h1) h2 (* 
h3 h3 h3) (* h5 h5 h5) j2) (* 256 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5)) (* 
460 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 2272 (* h1 h1 
h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 3992 (* h1 h1 h1) h2 (* h3 h3 
h3) (* h5 h5) h6 (* j2 j2)) (* 2912 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 j2
) (* 768 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6) (* 428 (* h1 h1 h1) h2 (* h3
 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 2022 (* h1 h1 h1) h2 (* h3 h3 h3) h5 
(* h6 h6) (* j2 j2 j2)) (* 3344 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) (* j2 
j2)) (* 2208 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) j2) (* 512 (* h1 h1 h1) 
h2 (* h3 h3 h3) h5 (* h6 h6)) (* 66 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 
(* j2 j2 j2 j2)) (* 316 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) 
(* 470 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 268 (* h1 h1 h1) 
h2 (* h3 h3) (* h4 h4 h4) h5 j2) (* 48 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5
) (* 186 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 820 
(* h1 h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 1296 (* h1 h1 h1)
 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2)) (* 844 (* h1 h1 h1) h2 (* h3 h3) 
(* h4 h4) (* h5 h5) j2) (* 176 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5)) 
(* 202 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 960 (* h1 
h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2)) (* 1458 (* h1 h1 h1) h2 (* h3 
h3) (* h4 h4) h5 h6 (* j2 j2)) (* 892 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6 
j2) (* 192 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6) (* 90 (* h1 h1 h1) h2 (* 
h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 448 (* h1 h1 h1) h2 (* h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2)) (* 770 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 
j2)) (* 568 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) j2) (* 144 (* h1 h1 h1) h2
 (* h3 h3) h4 (* h5 h5 h5)) (* 374 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2)) (* 1676 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) 
(* 2600 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 1696 (* h1 h1 h1
) h2 (* h3 h3) h4 (* h5 h5) h6 j2) (* 416 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5)
 h6) (* 206 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 972 
(* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2)) (* 1506 (* h1 h1 h1) h2 
(* h3 h3) h4 h5 (* h6 h6) (* j2 j2)) (* 980 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* 
h6 h6) j2) (* 240 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6)) (* 24 (* h1 h1 h1) 
h2 (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 64 (* h1 h1 h1) h2 (* h3 h3) (* h5
 h5 h5 h5) (* j2 j2)) (* 56 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5 h5) j2) (* 16 
(* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5 h5)) (* 96 (* h1 h1 h1) h2 (* h3 h3) (* h5
 h5 h5) h6 (* j2 j2 j2 j2)) (* 466 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 (* 
j2 j2 j2)) (* 774 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 564 
(* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 j2) (* 160 (* h1 h1 h1) h2 (* h3 h3) 
(* h5 h5 h5) h6) (* 182 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2)) (* 819 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
1265 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 840 (* h1 h1 h1
) h2 (* h3 h3) (* h5 h5) (* h6 h6) j2) (* 224 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5
) (* h6 h6)) (* 70 (* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) 
(* 328 (* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 518 (* h1 h1 
h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 356 (* h1 h1 h1) h2 (* h3 h3) h5 
(* h6 h6 h6) j2) (* 96 (* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6)) (* 15 (* h1 h1
 h1) h2 h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 61 (* h1 h1 h1) h2 h3 (* 
h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 67 (* h1 h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5
) (* j2 j2)) (* 21 (* h1 h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) j2) (- (* 2 (* h1 
h1 h1) h2 h3 (* h4 h4 h4) h5 h6 (* j2 j2 j2))) (- (* 4 (* h1 h1 h1) h2 h3 (* h4 
h4 h4) h5 h6 (* j2 j2))) (- (* 2 (* h1 h1 h1) h2 h3 (* h4 h4 h4) h5 h6 j2)) (* 
15 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 67 (* h1 h1 h1)
 h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 98 (* h1 h1 h1) h2 h3 (* h4 h4) 
(* h5 h5 h5) (* j2 j2)) (* 42 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) j2) (* 
51 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 216 (* h1 h1 h1
) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 277 (* h1 h1 h1) h2 h3 (* h4 h4)
 (* h5 h5) h6 (* j2 j2)) (* 136 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6 j2) 
(* 24 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6) (- (* 6 (* h1 h1 h1) h2 h3 (* 
h4 h4) h5 (* h6 h6) (* j2 j2 j2))) (- (* 12 (* h1 h1 h1) h2 h3 (* h4 h4) h5 (* 
h6 h6) (* j2 j2))) (- (* 6 (* h1 h1 h1) h2 h3 (* h4 h4) h5 (* h6 h6) j2)) (* 9 
(* h1 h1 h1) h2 h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 16 (* h1 h1 h1) h2 h3 h4 
(* h5 h5 h5 h5) (* j2 j2)) (* 7 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5 h5) j2) (* 36 
(* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 167 (* h1 h1 h1) h2 h3
 h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 269 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 
(* j2 j2)) (* 184 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 j2) (* 48 (* h1 h1 h1) 
h2 h3 h4 (* h5 h5 h5) h6) (* 57 (* h1 h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2)) (* 249 (* h1 h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 355
 (* h1 h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 211 (* h1 h1 h1) h2 h3 
h4 (* h5 h5) (* h6 h6) j2) (* 48 (* h1 h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6)) (- 
(* 6 (* h1 h1 h1) h2 h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2))) (- (* 12 (* h1 h1 h1) 
h2 h3 h4 h5 (* h6 h6 h6) (* j2 j2))) (- (* 6 (* h1 h1 h1) h2 h3 h4 h5 (* h6 h6 
h6) j2)) (* 8 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 24 (* h1 h1
 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 24 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5
) h6 j2) (* 8 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6) (* 21 (* h1 h1 h1) h2 h3 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 100 (* h1 h1 h1) h2 h3 (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2)) (* 171 (* h1 h1 h1) h2 h3 (* h5 h5 h5) (* h6 h6) (* j2 
j2)) (* 128 (* h1 h1 h1) h2 h3 (* h5 h5 h5) (* h6 h6) j2) (* 36 (* h1 h1 h1) h2 
h3 (* h5 h5 h5) (* h6 h6)) (* 21 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6) (* j2
 j2 j2 j2)) (* 94 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 145
 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 96 (* h1 h1 h1) h2 h3 
(* h5 h5) (* h6 h6 h6) j2) (* 24 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6)) (- 
(* 2 (* h1 h1 h1) h2 h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2))) (- (* 4 (* h1 h1 h1) 
h2 h3 h5 (* h6 h6 h6 h6) (* j2 j2))) (- (* 2 (* h1 h1 h1) h2 h3 h5 (* h6 h6 h6 
h6) j2)) (* (* h1 h1 h1) h2 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 2 (* h1 
h1 h1) h2 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* (* h1 h1 h1) h2 (* h4 h4 h4) 
(* h5 h5) h6 j2) (* 2 (* h1 h1 h1) h2 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) 
(* 6 (* h1 h1 h1) h2 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 4 (* h1 h1 h1) h2 
(* h4 h4) (* h5 h5 h5) h6 j2) (* 3 (* h1 h1 h1) h2 (* h4 h4) (* h5 h5) (* h6 h6)
 (* j2 j2 j2)) (* 7 (* h1 h1 h1) h2 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 
5 (* h1 h1 h1) h2 (* h4 h4) (* h5 h5) (* h6 h6) j2) (* (* h1 h1 h1) h2 (* h4 h4)
 (* h5 h5) (* h6 h6)) (* (* h1 h1 h1) h2 h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 
2 (* h1 h1 h1) h2 h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* (* h1 h1 h1) h2 h4 (* h5 
h5 h5 h5) h6 j2) (* 4 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) 
(* 13 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 13 (* h1 h1 h1) h2
 h4 (* h5 h5 h5) (* h6 h6) j2) (* 4 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6)) 
(* 3 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 8 (* h1 h1 h1) 
h2 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 7 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 
h6 h6) j2) (* 2 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6)) (* (* h1 h1 h1) h2 
(* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* 
h6 h6) (* j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6) j2) (* (* h1 h1
 h1) h2 (* h5 h5 h5 h5) (* h6 h6)) (* 2 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6
) (* j2 j2 j2)) (* 7 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 8 
(* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) j2) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5) 
(* h6 h6 h6)) (* (* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 3 
(* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 
h5) (* h6 h6 h6 h6) j2) (* (* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6)) (* 60 (* 
h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 396 (* h1 h1 h1) (* 
h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 720 (* h1 h1 h1) (* h3 h3 h3 h3) (* 
h4 h4) h5 (* j2 j2)) (* 384 (* h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 j2) (* 60 
(* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 396 (* h1 h1 h1) 
(* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 720 (* h1 h1 h1) (* h3 h3 h3 h3) 
h4 (* h5 h5) (* j2 j2)) (* 384 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) j2) (* 
140 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 1004 (* h1 h1 h1) 
(* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 2208 (* h1 h1 h1) (* h3 h3 h3 h3) h4 
h5 h6 (* j2 j2)) (* 1856 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 j2) (* 512 (* h1 
h1 h1) (* h3 h3 h3 h3) h4 h5 h6) (* 80 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6
 (* j2 j2 j2 j2)) (* 608 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2))
 (* 1488 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2)) (* 1472 (* h1 h1 
h1) (* h3 h3 h3 h3) (* h5 h5) h6 j2) (* 512 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 
h5) h6) (* 80 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 608 
(* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 1488 (* h1 h1 h1) (* 
h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 1472 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* 
h6 h6) j2) (* 512 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6)) (* 24 (* h1 h1 h1) 
(* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 168 (* h1 h1 h1) (* h3 h3 h3) 
(* h4 h4 h4) h5 (* j2 j2 j2)) (* 336 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 
(* j2 j2)) (* 192 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 j2) (* 60 (* h1 h1 
h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 426 (* h1 h1 h1) (* h3 
h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 888 (* h1 h1 h1) (* h3 h3 h3) (* h4 
h4) (* h5 h5) (* j2 j2)) (* 576 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) j2
) (* 80 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 592 (* h1 
h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2)) (* 1344 (* h1 h1 h1) (* h3 h3 
h3) (* h4 h4) h5 h6 (* j2 j2)) (* 1088 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6
 j2) (* 256 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6) (* 24 (* h1 h1 h1) (* h3 
h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 168 (* h1 h1 h1) (* h3 h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2)) (* 336 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 
j2)) (* 192 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) j2) (* 134 (* h1 h1 h1) 
(* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1029 (* h1 h1 h1) (* h3 h3 h3) 
h4 (* h5 h5) h6 (* j2 j2 j2)) (* 2532 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) h6 
(* j2 j2)) (* 2432 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) h6 j2) (* 768 (* h1 h1
 h1) (* h3 h3 h3) h4 (* h5 h5) h6) (* 88 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 
h6) (* j2 j2 j2 j2)) (* 680 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 
j2)) (* 1680 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2)) (* 1600 (* h1 
h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) j2) (* 512 (* h1 h1 h1) (* h3 h3 h3) h4 h5 
(* h6 h6)) (* 32 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 
256 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 672 (* h1 h1 h1) 
(* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 704 (* h1 h1 h1) (* h3 h3 h3) (* h5 
h5 h5) h6 j2) (* 256 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6) (* 72 (* h1 h1 
h1) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 580 (* h1 h1 h1) (* h3 
h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1548 (* h1 h1 h1) (* h3 h3 h3) (* h5
 h5) (* h6 h6) (* j2 j2)) (* 1680 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6) 
j2) (* 640 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6)) (* 32 (* h1 h1 h1) (* 
h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3) h5 
(* h6 h6 h6) (* j2 j2 j2)) (* 672 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* 
j2 j2)) (* 704 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) j2) (* 256 (* h1 h1 h1)
 (* h3 h3 h3) h5 (* h6 h6 h6)) (* 12 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2 j2)) (* 24 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2)) (* 12 
(* h1 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 j2) (* 9 (* h1 h1 h1) (* h3 h3) (* h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 75 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* 
h5 h5) (* j2 j2 j2)) (* 150 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 
j2)) (* 84 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 52 (* h1 h1 h1) 
(* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2)) (* 120 (* h1 h1 h1) (* h3 h3) (* h4 
h4 h4) h5 h6 (* j2 j2)) (* 84 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 h6 j2) (* 
16 (* h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 h6) (* 9 (* h1 h1 h1) (* h3 h3) (* h4 
h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 75 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2)) (* 150 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 
j2)) (* 84 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 30 (* h1 h1 h1) 
(* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 262 (* h1 h1 h1) (* h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 588 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2)) (* 468 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 j2) 
(* 112 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6) (* 84 (* h1 h1 h1) (* h3 
h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2)) (* 216 (* h1 h1 h1) (* h3 h3) (* h4 h4)
 h5 (* h6 h6) (* j2 j2)) (* 180 (* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) j2
) (* 48 (* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6)) (* 12 (* h1 h1 h1) (* h3 
h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 24 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5
 h5) (* j2 j2)) (* 12 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) j2) (* 21 (* h1 
h1 h1) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 187 (* h1 h1 h1) (* h3 
h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 438 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5 
h5) h6 (* j2 j2)) (* 384 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) h6 j2) (* 112 
(* h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) h6) (* 33 (* h1 h1 h1) (* h3 h3) h4 (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 299 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5) (* h6
 h6) (* j2 j2 j2)) (* 722 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2
)) (* 664 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 208 (* h1 h1 h1) 
(* h3 h3) h4 (* h5 h5) (* h6 h6)) (* 60 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 h6
) (* j2 j2 j2)) (* 168 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2)) (* 
156 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 h6) j2) (* 48 (* h1 h1 h1) (* h3 h3) 
h4 h5 (* h6 h6 h6)) (* 16 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2)
) (* 48 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 48 (* h1 h1 h1) 
(* h3 h3) (* h5 h5 h5 h5) h6 j2) (* 16 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6
) (* 12 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 112 
(* h1 h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 284 (* h1 h1 h1) 
(* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 280 (* h1 h1 h1) (* h3 h3) (* h5 
h5 h5) (* h6 h6) j2) (* 96 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6)) (* 12 
(* h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 112 (* h1 h1 
h1) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 284 (* h1 h1 h1) (* h3 h3)
 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 280 (* h1 h1 h1) (* h3 h3) (* h5 h5) (* h6
 h6 h6) j2) (* 96 (* h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6)) (* 16 (* h1 h1 
h1) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2)) (* 48 (* h1 h1 h1) (* h3 h3) h5 
(* h6 h6 h6 h6) (* j2 j2)) (* 48 (* h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6 h6) j2) 
(* 16 (* h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6 h6)) (* 3 (* h1 h1 h1) h3 (* h4 h4 
h4 h4) (* h5 h5) (* j2 j2 j2)) (* 3 (* h1 h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5) 
(* j2 j2)) (* 6 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 12 
(* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 13 (* h1 h1 h1) h3 (* h4
 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 20 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) 
h6 (* j2 j2)) (* 7 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 j2) (* 3 (* h1 h1 
h1) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 3 (* h1 h1 h1) h3 (* h4 h4) 
(* h5 h5 h5 h5) (* j2 j2)) (* 20 (* h1 h1 h1) h3 (* h4 h4) (* h5 h5 h5) h6 (* j2
 j2 j2)) (* 51 (* h1 h1 h1) h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 28 (* h1 
h1 h1) h3 (* h4 h4) (* h5 h5 h5) h6 j2) (* 21 (* h1 h1 h1) h3 (* h4 h4) (* h5 h5
) (* h6 h6) (* j2 j2 j2)) (* 43 (* h1 h1 h1) h3 (* h4 h4) (* h5 h5) (* h6 h6) 
(* j2 j2)) (* 26 (* h1 h1 h1) h3 (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 4 (* h1 h1
 h1) h3 (* h4 h4) (* h5 h5) (* h6 h6)) (* 7 (* h1 h1 h1) h3 h4 (* h5 h5 h5 h5) 
h6 (* j2 j2 j2)) (* 14 (* h1 h1 h1) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 7 (* 
h1 h1 h1) h3 h4 (* h5 h5 h5 h5) h6 j2) (* 22 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2)) (* 67 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) 
(* 61 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6) j2) (* 16 (* h1 h1 h1) h3 h4 (* 
h5 h5 h5) (* h6 h6)) (* 15 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
)) (* 38 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 31 (* h1 h1 h1)
 h3 h4 (* h5 h5) (* h6 h6 h6) j2) (* 8 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6)
) (* 4 (* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 12 (* h1 h1 h1
) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 12 (* h1 h1 h1) h3 (* h5 h5 h5 h5) 
(* h6 h6) j2) (* 4 (* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6)) (* 8 (* h1 h1 h1) 
h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 28 (* h1 h1 h1) h3 (* h5 h5 h5) 
(* h6 h6 h6) (* j2 j2)) (* 32 (* h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6 h6) j2) (* 
12 (* h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6 h6)) (* 4 (* h1 h1 h1) h3 (* h5 h5) (* 
h6 h6 h6 h6) (* j2 j2 j2)) (* 12 (* h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2
 j2)) (* 12 (* h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6 h6) j2) (* 4 (* h1 h1 h1) h3 
(* h5 h5) (* h6 h6 h6 h6)) (* 216 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* 
j2 j2 j2 j2)) (* 648 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2)) (* 
720 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2)) (* 352 (* h1 h1) (* h2 
h2 h2 h2) (* h3 h3 h3) h5 j2) (* 64 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5) 
(* 108 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 360 (* h1 
h1) (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2)) (* 444 (* h1 h1) (* h2 h2 h2 
h2) (* h3 h3) h4 h5 (* j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h4 h5 
j2) (* 48 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h4 h5) (* 72 (* h1 h1) (* h2 h2 h2
 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2 h2) (* h3
 h3) (* h5 h5) (* j2 j2 j2)) (* 296 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5
) (* j2 j2)) (* 160 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) j2) (* 32 (* 
h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5)) (* 72 (* h1 h1) (* h2 h2 h2 h2) (* 
h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 
(* j2 j2 j2)) (* 296 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2)) (* 160
 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 j2) (* 32 (* h1 h1) (* h2 h2 h2 h2) 
(* h3 h3) h5 h6) (* 18 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2)
) (* 66 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2)) (* 90 (* h1 h1) 
(* h2 h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2)) (* 54 (* h1 h1) (* h2 h2 h2 h2) h3 h4
 (* h5 h5) j2) (* 12 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5)) (* 6 (* h1 h1) 
(* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 22 (* h1 h1) (* h2 h2 h2 h2)
 h3 (* h5 h5 h5) (* j2 j2 j2)) (* 30 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) 
(* j2 j2)) (* 18 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) j2) (* 4 (* h1 h1) 
(* h2 h2 h2 h2) h3 (* h5 h5 h5)) (* 12 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) h6
 (* j2 j2 j2 j2)) (* 44 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2)) 
(* 60 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) h6 (* j2 j2)) (* 36 (* h1 h1) (* h2
 h2 h2 h2) h3 (* h5 h5) h6 j2) (* 8 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5) h6) 
(* 2 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 8 (* h1 h1) 
(* h2 h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2 h2 h2) h4
 (* h5 h5) h6 (* j2 j2)) (* 8 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 j2) (* 2
 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6) (* (* h1 h1) (* h2 h2 h2 h2) (* h5 
h5 h5) h6 (* j2 j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 
j2 j2)) (* 6 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2)) (* 4 (* h1 h1)
 (* h2 h2 h2 h2) (* h5 h5 h5) h6 j2) (* (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) 
h6) (* (* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 4 (* h1
 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2 
h2 h2) (* h5 h5) (* h6 h6) (* j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5) 
(* h6 h6) j2) (* (* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6)) (* 216 (* h1 h1)
 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 1296 (* h1 h1) (* h2 h2 h2)
 (* h3 h3 h3 h3) h5 (* j2 j2 j2)) (* 2016 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3 h3)
 h5 (* j2 j2)) (* 1216 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3 h3) h5 j2) (* 256 (* 
h1 h1) (* h2 h2 h2) (* h3 h3 h3 h3) h5) (* 1188 (* h1 h1) (* h2 h2 h2) (* h3 h3 
h3) h4 h5 (* j2 j2 j2 j2)) (* 4356 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* 
j2 j2 j2)) (* 5808 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2)) (* 3344 
(* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 j2) (* 704 (* h1 h1) (* h2 h2 h2) (* 
h3 h3 h3) h4 h5) (* 612 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 
j2 j2)) (* 2292 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 
3104 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2)) (* 1808 (* h1 h1) 
(* h2 h2 h2) (* h3 h3 h3) (* h5 h5) j2) (* 384 (* h1 h1) (* h2 h2 h2) (* h3 h3 
h3) (* h5 h5)) (* 612 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2))
 (* 2292 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2)) (* 3104 (* h1 
h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2)) (* 1808 (* h1 h1) (* h2 h2 h2) 
(* h3 h3 h3) h5 h6 j2) (* 384 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6) (* 264 
(* h1 h1) (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 1088 (* h1 h1)
 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 1568 (* h1 h1) (* h2 h2 h2
) (* h3 h3) (* h4 h4) h5 (* j2 j2)) (* 952 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* 
h4 h4) h5 j2) (* 208 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h4 h4) h5) (* 396 (* h1
 h1) (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 1488 (* h1 h1) (* 
h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 2076 (* h1 h1) (* h2 h2 h2) 
(* h3 h3) h4 (* h5 h5) (* j2 j2)) (* 1272 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 
(* h5 h5) j2) (* 288 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 (* h5 h5)) (* 420 (* h1
 h1) (* h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 1744 (* h1 h1) (* h2 h2
 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 2524 (* h1 h1) (* h2 h2 h2) (* h3 h3) 
h4 h5 h6 (* j2 j2)) (* 1536 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 h6 j2) (* 336
 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 h6) (* 108 (* h1 h1) (* h2 h2 h2) (* h3 
h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 408 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 
h5 h5) (* j2 j2 j2)) (* 572 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 
j2)) (* 352 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) j2) (* 80 (* h1 h1) 
(* h2 h2 h2) (* h3 h3) (* h5 h5 h5)) (* 264 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* 
h5 h5) h6 (* j2 j2 j2 j2)) (* 1040 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6
 (* j2 j2 j2)) (* 1488 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2)) 
(* 920 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 j2) (* 208 (* h1 h1) (* h2 
h2 h2) (* h3 h3) (* h5 h5) h6) (* 156 (* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 
h6) (* j2 j2 j2 j2)) (* 656 (* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 
j2 j2)) (* 956 (* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2)) (* 584 
(* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) j2) (* 128 (* h1 h1) (* h2 h2 h2) 
(* h3 h3) h5 (* h6 h6)) (* 48 (* h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2)) (* 196 (* h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2))
 (* 292 (* h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2)) (* 188 (* h1 h1
) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) j2) (* 44 (* h1 h1) (* h2 h2 h2) h3 (* h4 
h4) (* h5 h5)) (- (* 2 (* h1 h1) (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2)
)) (- (* 6 (* h1 h1) (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2))) (- (* 6 (* 
h1 h1) (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2))) (- (* 2 (* h1 h1) (* h2 h2 h2
) h3 (* h4 h4) h5 h6 j2)) (* 33 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 
j2 j2 j2)) (* 137 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2)) (* 207
 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2)) (* 135 (* h1 h1) (* h2 h2 
h2) h3 h4 (* h5 h5 h5) j2) (* 32 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5 h5)) (* 
85 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 335 (* h1 h1) 
(* h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2)) (* 495 (* h1 h1) (* h2 h2 h2) h3 
h4 (* h5 h5) h6 (* j2 j2)) (* 325 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) h6 j2) 
(* 80 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) h6) (- (* 4 (* h1 h1) (* h2 h2 h2) 
h3 h4 h5 (* h6 h6) (* j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2 h2) h3 h4 h5 
(* h6 h6) (* j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) (* 
j2 j2))) (- (* 4 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) j2)) (* 6 (* h1 h1) 
(* h2 h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 16 (* h1 h1) (* h2 h2 h2) h3 
(* h5 h5 h5 h5) (* j2 j2)) (* 14 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5) j2) 
(* 4 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5)) (* 34 (* h1 h1) (* h2 h2 h2) h3 
(* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 134 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) 
h6 (* j2 j2 j2)) (* 198 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2)) (* 
130 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 j2) (* 32 (* h1 h1) (* h2 h2 h2) 
h3 (* h5 h5 h5) h6) (* 33 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2
 j2 j2)) (* 129 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
191 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2)) (* 127 (* h1 h1) 
(* h2 h2 h2) h3 (* h5 h5) (* h6 h6) j2) (* 32 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5
) (* h6 h6)) (- (* 2 (* h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2)))
 (- (* 6 (* h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2))) (- (* 6 (* h1 
h1) (* h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2))) (- (* 2 (* h1 h1) (* h2 h2 h2) 
h3 h5 (* h6 h6 h6) j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2
 j2 j2 j2)) (* 14 (* h1 h1) (* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) 
(* 24 (* h1 h1) (* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 18 (* h1 h1) 
(* h2 h2 h2) (* h4 h4) (* h5 h5) h6 j2) (* 5 (* h1 h1) (* h2 h2 h2) (* h4 h4) 
(* h5 h5) h6) (* 3 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 14 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 24 (* h1 h1) 
(* h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2)) (* 18 (* h1 h1) (* h2 h2 h2) h4 (* h5
 h5 h5) h6 j2) (* 5 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5 h5) h6) (* 6 (* h1 h1) 
(* h2 h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 27 (* h1 h1) (* h2 h2 h2
) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 45 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5)
 (* h6 h6) (* j2 j2)) (* 33 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5) (* h6 h6) j2) 
(* 9 (* h1 h1) (* h2 h2 h2) h4 (* h5 h5) (* h6 h6)) (* (* h1 h1) (* h2 h2 h2) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5 h5) h6 
(* j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5 h5) h6 j2) (* (* h1 h1) (* h2
 h2 h2) (* h5 h5 h5 h5) h6) (* 3 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2)) (* 13 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2
)) (* 21 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 15 (* h1 h1
) (* h2 h2 h2) (* h5 h5 h5) (* h6 h6) j2) (* 4 (* h1 h1) (* h2 h2 h2) (* h5 h5 
h5) (* h6 h6)) (* 3 (* h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
)) (* 13 (* h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 21 (* h1
 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 15 (* h1 h1) (* h2 h2 h2)
 (* h5 h5) (* h6 h6 h6) j2) (* 4 (* h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6 h6)) 
(* 648 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 4320 (* h1 
h1) (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 8352 (* h1 h1) (* h2 h2) 
(* h3 h3 h3 h3) h4 h5 (* j2 j2)) (* 6144 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 
h5 j2) (* 1536 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 h5) (* 324 (* h1 h1) (* h2
 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 2160 (* h1 h1) (* h2 h2) (* 
h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 4176 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3)
 (* h5 h5) (* j2 j2)) (* 3072 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) j2) 
(* 768 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5)) (* 324 (* h1 h1) (* h2 h2)
 (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 2160 (* h1 h1) (* h2 h2) (* h3 h3 h3 
h3) h5 h6 (* j2 j2 j2)) (* 4176 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 
j2)) (* 3072 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 h6 j2) (* 768 (* h1 h1) (* 
h2 h2) (* h3 h3 h3 h3) h5 h6) (* 984 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) 
h5 (* j2 j2 j2 j2)) (* 4496 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 
j2 j2)) (* 7456 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2)) (* 5088
 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 j2) (* 1216 (* h1 h1) (* h2 h2) 
(* h3 h3 h3) (* h4 h4) h5) (* 1080 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5)
 (* j2 j2 j2 j2)) (* 5136 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2
 j2)) (* 8752 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 6272 
(* h1 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) j2) (* 1600 (* h1 h1) (* h2 h2) 
(* h3 h3 h3) h4 (* h5 h5)) (* 1860 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6 (* 
j2 j2 j2 j2)) (* 8236 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) 
(* 13328 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 8944 (* h1 h1) 
(* h2 h2) (* h3 h3 h3) h4 h5 h6 j2) (* 2112 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 
h5 h6) (* 126 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 
870 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2)) (* 1796 (* h1 h1
) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2)) (* 1424 (* h1 h1) (* h2 h2) (* 
h3 h3 h3) (* h5 h5 h5) j2) (* 384 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5))
 (* 942 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 4198 
(* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 6844 (* h1 h1) 
(* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2)) (* 4704 (* h1 h1) (* h2 h2) (* h3
 h3 h3) (* h5 h5) h6 j2) (* 1152 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6) 
(* 876 (* h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 3740 
(* h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 5872 (* h1 h1) 
(* h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 3856 (* h1 h1) (* h2 h2) (* h3
 h3 h3) h5 (* h6 h6) j2) (* 896 (* h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* h6 h6)) 
(* 132 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 662 (* 
h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 1042 (* h1 h1) (* h2
 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 656 (* h1 h1) (* h2 h2) (* h3 h3) 
(* h4 h4 h4) h5 j2) (* 144 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5) (* 366
 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 1626 (* 
h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 2668 (* h1 h1) 
(* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2)) (* 1892 (* h1 h1) (* h2 h2) 
(* h3 h3) (* h4 h4) (* h5 h5) j2) (* 480 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4)
 (* h5 h5)) (* 396 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)
) (* 1930 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2)) (* 3002 
(* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 1884 (* h1 h1) (* h2
 h2) (* h3 h3) (* h4 h4) h5 h6 j2) (* 416 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4
) h5 h6) (* 186 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) 
(* 910 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 1528 (* h1
 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 1104 (* h1 h1) (* h2 h2) 
(* h3 h3) h4 (* h5 h5 h5) j2) (* 288 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5 
h5)) (* 711 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 
3076 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 4805 (* h1 
h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 3280 (* h1 h1) (* h2 h2) 
(* h3 h3) h4 (* h5 h5) h6 j2) (* 832 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) 
h6) (* 396 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 
1874 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2)) (* 2878 (* h1 
h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2)) (* 1800 (* h1 h1) (* h2 h2) 
(* h3 h3) h4 h5 (* h6 h6) j2) (* 400 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 
h6)) (* 66 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 146 
(* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2)) (* 116 (* h1 h1) (* h2 
h2) (* h3 h3) (* h5 h5 h5 h5) j2) (* 32 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 
h5 h5)) (* 189 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 848 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 1335 (* h1
 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 924 (* h1 h1) (* h2 h2) 
(* h3 h3) (* h5 h5 h5) h6 j2) (* 240 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) 
h6) (* 339 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 1426 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 2143 
(* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 1412 (* h1 h1) 
(* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) j2) (* 352 (* h1 h1) (* h2 h2) (* h3 h3)
 (* h5 h5) (* h6 h6)) (* 132 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2
 j2 j2 j2)) (* 606 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) 
(* 918 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 572 (* h1 h1)
 (* h2 h2) (* h3 h3) h5 (* h6 h6 h6) j2) (* 128 (* h1 h1) (* h2 h2) (* h3 h3) h5
 (* h6 h6 h6)) (* 27 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 
j2)) (* 123 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 175 
(* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2)) (* 95 (* h1 h1) (* h2 
h2) h3 (* h4 h4 h4) (* h5 h5) j2) (* 16 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* 
h5 h5)) (- (* 2 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) h5 h6 (* j2 j2 j2))) (- (* 4
 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) h5 h6 (* j2 j2))) (- (* 2 (* h1 h1) (* h2 
h2) h3 (* h4 h4 h4) h5 h6 j2)) (* 27 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5 
h5) (* j2 j2 j2 j2)) (* 135 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 
j2 j2)) (* 237 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 173 
(* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) j2) (* 44 (* h1 h1) (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5)) (* 90 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* 
j2 j2 j2 j2)) (* 398 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2))
 (* 566 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 326 (* h1 h1
) (* h2 h2) h3 (* h4 h4) (* h5 h5) h6 j2) (* 68 (* h1 h1) (* h2 h2) h3 (* h4 h4)
 (* h5 h5) h6) (- (* 6 (* h1 h1) (* h2 h2) h3 (* h4 h4) h5 (* h6 h6) (* j2 j2 j2
))) (- (* 12 (* h1 h1) (* h2 h2) h3 (* h4 h4) h5 (* h6 h6) (* j2 j2))) (- (* 6 
(* h1 h1) (* h2 h2) h3 (* h4 h4) h5 (* h6 h6) j2)) (* 21 (* h1 h1) (* h2 h2) h3 
h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 54 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5 h5)
 (* j2 j2)) (* 45 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5 h5) j2) (* 12 (* h1 h1) 
(* h2 h2) h3 h4 (* h5 h5 h5 h5)) (* 63 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6
 (* j2 j2 j2 j2)) (* 299 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2))
 (* 497 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2)) (* 357 (* h1 h1) 
(* h2 h2) h3 h4 (* h5 h5 h5) h6 j2) (* 96 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5)
 h6) (* 99 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 424
 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 601 (* h1 h1) 
(* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 360 (* h1 h1) (* h2 h2) h3 h4 
(* h5 h5) (* h6 h6) j2) (* 84 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6)) (- 
(* 6 (* h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2))) (- (* 12 (* h1 h1)
 (* h2 h2) h3 h4 h5 (* h6 h6 h6) (* j2 j2))) (- (* 6 (* h1 h1) (* h2 h2) h3 h4 
h5 (* h6 h6 h6) j2)) (* 21 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2
)) (* 56 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 51 (* h1 h1) 
(* h2 h2) h3 (* h5 h5 h5 h5) h6 j2) (* 16 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5)
 h6) (* 36 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 161
 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 254 (* h1 h1) 
(* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 177 (* h1 h1) (* h2 h2) h3 (* 
h5 h5 h5) (* h6 h6) j2) (* 48 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6)) (* 
36 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 149 (* h1 
h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 210 (* h1 h1) (* h2 h2)
 h3 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 129 (* h1 h1) (* h2 h2) h3 (* h5 h5) 
(* h6 h6 h6) j2) (* 32 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6)) (- (* 2 
(* h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2))) (- (* 4 (* h1 h1) (* h2
 h2) h3 h5 (* h6 h6 h6 h6) (* j2 j2))) (- (* 2 (* h1 h1) (* h2 h2) h3 h5 (* h6 
h6 h6 h6) j2)) (* 2 (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) 
(* 5 (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 4 (* h1 h1) (* 
h2 h2) (* h4 h4 h4) (* h5 h5) h6 j2) (* (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 
h5) h6) (* 4 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 13 
(* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 14 (* h1 h1) (* h2 
h2) (* h4 h4) (* h5 h5 h5) h6 j2) (* 5 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5
) h6) (* 6 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 15
 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 12 (* h1 h1) 
(* h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 3 (* h1 h1) (* h2 h2) (* h4 h4) 
(* h5 h5) (* h6 h6)) (* 2 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)
) (* 6 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 6 (* h1 h1) (* h2
 h2) h4 (* h5 h5 h5 h5) h6 j2) (* 2 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5 h5) h6) 
(* 8 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 25 (* h1 h1)
 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 26 (* h1 h1) (* h2 h2) h4 (* 
h5 h5 h5) (* h6 h6) j2) (* 9 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5) (* h6 h6)) (* 6
 (* h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 15 (* h1 h1) (* 
h2 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 12 (* h1 h1) (* h2 h2) h4 (* h5 
h5) (* h6 h6 h6) j2) (* 3 (* h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6 h6)) (* 2 
(* h1 h1) (* h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 6 (* h1 h1) (* h2
 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 6 (* h1 h1) (* h2 h2) (* h5 h5 h5 
h5) (* h6 h6) j2) (* 2 (* h1 h1) (* h2 h2) (* h5 h5 h5 h5) (* h6 h6)) (* 4 (* h1
 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2)
 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 12 (* h1 h1) (* h2 h2) (* h5 h5 h5) (* 
h6 h6 h6) j2) (* 4 (* h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6 h6)) (* 2 (* h1 h1)
 (* h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 5 (* h1 h1) (* h2 h2) (* 
h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 4 (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6
 h6) j2) (* (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6 h6)) (* 324 (* h1 h1) h2 
(* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 2232 (* h1 h1) h2 (* h3 h3 h3 
h3) (* h4 h4) h5 (* j2 j2 j2)) (* 4512 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5
 (* j2 j2)) (* 3264 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 j2) (* 768 (* h1 
h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5) (* 342 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* 
h5 h5) (* j2 j2 j2 j2)) (* 2460 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 
j2 j2)) (* 5520 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 4992 (* 
h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) j2) (* 1536 (* h1 h1) h2 (* h3 h3 h3 h3) 
h4 (* h5 h5)) (* 648 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 
4464 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 9024 (* h1 h1) h2 
(* h3 h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 6528 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 
h6 j2) (* 1536 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6) (* 24 (* h1 h1) h2 (* h3 
h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 232 (* h1 h1) h2 (* h3 h3 h3 h3) (* 
h5 h5 h5) (* j2 j2 j2)) (* 720 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 
j2)) (* 768 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5 h5) j2) (* 256 (* h1 h1) h2 
(* h3 h3 h3 h3) (* h5 h5 h5)) (* 330 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2)) (* 2308 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2))
 (* 4848 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2)) (* 3840 (* h1 h1) 
h2 (* h3 h3 h3 h3) (* h5 h5) h6 j2) (* 1024 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 
h5) h6) (* 324 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 
2232 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 4512 (* h1 h1) 
h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 3264 (* h1 h1) h2 (* h3 h3 h3 h3) 
h5 (* h6 h6) j2) (* 768 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6)) (* 108 (* h1 
h1) h2 (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 794 (* h1 h1) h2 (* h3 
h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 1764 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 
h4) h5 (* j2 j2)) (* 1344 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) h5 j2) (* 320 
(* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) h5) (* 276 (* h1 h1) h2 (* h3 h3 h3) (* 
h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 2024 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2)) (* 4652 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) 
(* j2 j2)) (* 4192 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) j2) (* 1216 (* 
h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5)) (* 330 (* h1 h1) h2 (* h3 h3 h3) (* 
h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 2420 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 
(* j2 j2 j2)) (* 5372 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 
4144 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 j2) (* 1024 (* h1 h1) h2 (* h3 h3
 h3) (* h4 h4) h5 h6) (* 129 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 
j2 j2)) (* 977 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 2364 
(* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 2256 (* h1 h1) h2 (* h3 
h3 h3) h4 (* h5 h5 h5) j2) (* 704 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5 h5)) (* 
551 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 4037 (* h1 h1)
 h2 (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 9204 (* h1 h1) h2 (* h3 h3 h3)
 h4 (* h5 h5) h6 (* j2 j2)) (* 8096 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 j2
) (* 2432 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6) (* 336 (* h1 h1) h2 (* h3 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 2458 (* h1 h1) h2 (* h3 h3 h3) h4 h5 
(* h6 h6) (* j2 j2 j2)) (* 5452 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 
j2)) (* 4256 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) j2) (* 1088 (* h1 h1) h2 
(* h3 h3 h3) h4 h5 (* h6 h6)) (* 24 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5 h5) 
(* j2 j2 j2)) (* 136 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2)) (* 176
 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5 h5) j2) (* 64 (* h1 h1) h2 (* h3 h3 h3) 
(* h5 h5 h5 h5)) (* 136 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2
)) (* 1010 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 2348 (* h1
 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 2096 (* h1 h1) h2 (* h3 h3 h3
) (* h5 h5 h5) h6 j2) (* 640 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6) (* 271 
(* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 1979 (* h1 h1)
 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 4468 (* h1 h1) h2 (* h3 h3
 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 3840 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) 
(* h6 h6) j2) (* 1152 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6)) (* 114 (* 
h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 832 (* h1 h1) h2 (* 
h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 1844 (* h1 h1) h2 (* h3 h3 h3) h5 (* 
h6 h6 h6) (* j2 j2)) (* 1456 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) j2) (* 
384 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6)) (* 54 (* h1 h1) h2 (* h3 h3) (* 
h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 118 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2)) (* 80 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4 h4) h5 j2) (* 16 (* h1 h1) 
h2 (* h3 h3) (* h4 h4 h4 h4) h5) (* 36 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) (* h5
 h5) (* j2 j2 j2 j2)) (* 327 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2
 j2 j2)) (* 759 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2)) (* 612 
(* h1 h1) h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 144 (* h1 h1) h2 (* h3 h3) 
(* h4 h4 h4) (* h5 h5)) (* 218 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 
j2 j2)) (* 486 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2)) (* 348 (* h1
 h1) h2 (* h3 h3) (* h4 h4 h4) h5 h6 j2) (* 80 (* h1 h1) h2 (* h3 h3) (* h4 h4 
h4) h5 h6) (* 36 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) 
(* 339 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 821 (* h1 
h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 744 (* h1 h1) h2 (* h3 h3)
 (* h4 h4) (* h5 h5 h5) j2) (* 208 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5)
) (* 117 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1039 
(* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 2366 (* h1 h1) h2
 (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 1920 (* h1 h1) h2 (* h3 h3) (* 
h4 h4) (* h5 h5) h6 j2) (* 512 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) h6) 
(* 330 (* h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2)) (* 750 (* h1 
h1) h2 (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 564 (* h1 h1) h2 (* h3 h3)
 (* h4 h4) h5 (* h6 h6) j2) (* 144 (* h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* h6 h6)
) (* 69 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 183 (* h1 h1)
 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 168 (* h1 h1) h2 (* h3 h3) h4 (* 
h5 h5 h5 h5) j2) (* 48 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5)) (* 81 (* h1 h1
) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 736 (* h1 h1) h2 (* h3 h3)
 h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 1719 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5)
 h6 (* j2 j2)) (* 1516 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 j2) (* 464 (* 
h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6) (* 126 (* h1 h1) h2 (* h3 h3) h4 (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2)) (* 1094 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6
 h6) (* j2 j2 j2)) (* 2440 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 
j2)) (* 1988 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 576 (* h1 h1) 
h2 (* h3 h3) h4 (* h5 h5) (* h6 h6)) (* 222 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 
h6 h6) (* j2 j2 j2)) (* 514 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2))
 (* 404 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) j2) (* 112 (* h1 h1) h2 (* h3 
h3) h4 h5 (* h6 h6 h6)) (* 72 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2
 j2)) (* 200 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 192 (* h1 
h1) h2 (* h3 h3) (* h5 h5 h5 h5) h6 j2) (* 64 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5
 h5) h6) (* 45 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 394 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 883 (* h1 
h1) h2 (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 756 (* h1 h1) h2 (* h3 h3)
 (* h5 h5 h5) (* h6 h6) j2) (* 240 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* h6 h6)
) (* 45 (* h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 382 
(* h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 833 (* h1 h1) h2 
(* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 680 (* h1 h1) h2 (* h3 h3) (* h5 
h5) (* h6 h6 h6) j2) (* 208 (* h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6 h6)) (* 56
 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2)) (* 132 (* h1 h1) h2 (* 
h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2)) (* 108 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 
h6 h6) j2) (* 32 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6 h6)) (* 12 (* h1 h1) h2 
h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 19 (* h1 h1) h2 h3 (* h4 h4 h4 h4)
 (* h5 h5) (* j2 j2)) (* 7 (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) j2) (* 24 
(* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 65 (* h1 h1) h2 h3 
(* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 35 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5
 h5) j2) (* 52 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 93 (* 
h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 49 (* h1 h1) h2 h3 (* h4 h4
 h4) (* h5 h5) h6 j2) (* 8 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6) (* 12 (* 
h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 28 (* h1 h1) h2 h3 (* h4
 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 14 (* h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5 h5) 
j2) (* 80 (* h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 225 (* h1 
h1) h2 h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 171 (* h1 h1) h2 h3 (* h4 h4) 
(* h5 h5 h5) h6 j2) (* 40 (* h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) h6) (* 84 (* h1
 h1) h2 h3 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 165 (* h1 h1) h2 h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 105 (* h1 h1) h2 h3 (* h4 h4) (* h5 
h5) (* h6 h6) j2) (* 24 (* h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* h6 h6)) (* 28 (* 
h1 h1) h2 h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 74 (* h1 h1) h2 h3 h4 (* h5 
h5 h5 h5) h6 (* j2 j2)) (* 60 (* h1 h1) h2 h3 h4 (* h5 h5 h5 h5) h6 j2) (* 16 
(* h1 h1) h2 h3 h4 (* h5 h5 h5 h5) h6) (* 88 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2)) (* 253 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2))
 (* 227 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) j2) (* 72 (* h1 h1) h2 h3 h4 
(* h5 h5 h5) (* h6 h6)) (* 60 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2
 j2)) (* 127 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 91 (* h1 h1
) h2 h3 h4 (* h5 h5) (* h6 h6 h6) j2) (* 24 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 
h6 h6)) (* 16 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 46 (* 
h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 46 (* h1 h1) h2 h3 (* h5 h5
 h5 h5) (* h6 h6) j2) (* 16 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6)) (* 32 (* 
h1 h1) h2 h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 93 (* h1 h1) h2 h3 (* h5
 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 91 (* h1 h1) h2 h3 (* h5 h5 h5) (* h6 h6 h6) 
j2) (* 32 (* h1 h1) h2 h3 (* h5 h5 h5) (* h6 h6 h6)) (* 16 (* h1 h1) h2 h3 (* h5
 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 36 (* h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6 
h6) (* j2 j2)) (* 28 (* h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6 h6) j2) (* 8 (* h1 h1
) h2 h3 (* h5 h5) (* h6 h6 h6 h6)) (* (* h1 h1) h2 (* h4 h4 h4) (* h5 h5 h5) h6 
(* j2 j2)) (* (* h1 h1) h2 (* h4 h4 h4) (* h5 h5 h5) h6 j2) (* (* h1 h1) h2 (* 
h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2)) (* (* h1 h1) h2 (* h4 h4) (* h5 h5 h5 h5) 
h6 j2) (* 3 (* h1 h1) h2 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 4 (* h1 
h1) h2 (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* (* h1 h1) h2 (* h4 h4) (* h5 h5 
h5) (* h6 h6)) (* 2 (* h1 h1) h2 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 3 
(* h1 h1) h2 h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* (* h1 h1) h2 h4 (* h5 h5 h5 h5)
 (* h6 h6)) (* 3 (* h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 5 (* h1
 h1) h2 h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 2 (* h1 h1) h2 h4 (* h5 h5 h5) (* h6
 h6 h6)) (* (* h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 2 (* h1 h1) 
h2 (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* (* h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6 h6)
) (* (* h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 2 (* h1 h1) h2 (* 
h5 h5 h5) (* h6 h6 h6 h6) j2) (* (* h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6 h6)) (* 
12 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 108 (* h1 h1) 
(* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 288 (* h1 h1) (* h3 h3 h3 h3) 
(* h4 h4 h4) h5 (* j2 j2)) (* 192 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 j2) 
(* 30 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 288 (* 
h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 864 (* h1 h1) (* h3 
h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2)) (* 768 (* h1 h1) (* h3 h3 h3 h3) (* h4 
h4) (* h5 h5) j2) (* 40 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2
)) (* 376 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2)) (* 1104 (* h1 
h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 1024 (* h1 h1) (* h3 h3 h3 h3)
 (* h4 h4) h5 h6 j2) (* 256 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6) (* 12 (* 
h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 108 (* h1 h1) (* h3 
h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 288 (* h1 h1) (* h3 h3 h3 h3) h4 (* 
h5 h5 h5) (* j2 j2)) (* 192 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5 h5) j2) (* 67 
(* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 676 (* h1 h1) (* 
h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 2256 (* h1 h1) (* h3 h3 h3 h3) h4 
(* h5 h5) h6 (* j2 j2)) (* 2752 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 j2) 
(* 1024 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6) (* 44 (* h1 h1) (* h3 h3 h3 
h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 428 (* h1 h1) (* h3 h3 h3 h3) h4 h5 (* 
h6 h6) (* j2 j2 j2)) (* 1344 (* h1 h1) (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2)
) (* 1472 (* h1 h1) (* h3 h3 h3 h3) h4 h5 (* h6 h6) j2) (* 512 (* h1 h1) (* h3 
h3 h3 h3) h4 h5 (* h6 h6)) (* 16 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2)) (* 160 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 
528 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 640 (* h1 h1) (* h3 
h3 h3 h3) (* h5 h5 h5) h6 j2) (* 256 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6) 
(* 36 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 372 (* 
h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1296 (* h1 h1) (* h3
 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 1728 (* h1 h1) (* h3 h3 h3 h3) (* 
h5 h5) (* h6 h6) j2) (* 768 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6)) (* 16
 (* h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 160 (* h1 h1) (* 
h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 528 (* h1 h1) (* h3 h3 h3 h3) h5 
(* h6 h6 h6) (* j2 j2)) (* 640 (* h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6 h6) j2) (* 
256 (* h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6 h6)) (* 12 (* h1 h1) (* h3 h3 h3) (* 
h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 60 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2)) (* 48 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5 j2) (* 9 (* h1 h1) 
(* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 102 (* h1 h1) (* h3 h3 
h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 348 (* h1 h1) (* h3 h3 h3) (* h4 h4 
h4) (* h5 h5) (* j2 j2)) (* 336 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) j2
) (* 52 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2)) (* 276 (* h1 h1)
 (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2)) (* 288 (* h1 h1) (* h3 h3 h3) (* h4 
h4 h4) h5 h6 j2) (* 64 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 h6) (* 9 (* h1 h1)
 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 102 (* h1 h1) (* h3 h3 
h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 348 (* h1 h1) (* h3 h3 h3) (* h4 h4)
 (* h5 h5 h5) (* j2 j2)) (* 336 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) j2
) (* 30 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 349 
(* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 1272 (* h1 h1) 
(* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 1536 (* h1 h1) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 j2) (* 448 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6)
 (* 84 (* h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2)) (* 468 (* h1 
h1) (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 576 (* h1 h1) (* h3 h3 h3)
 (* h4 h4) h5 (* h6 h6) j2) (* 192 (* h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* h6 h6)
) (* 12 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 60 (* h1 h1) 
(* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 48 (* h1 h1) (* h3 h3 h3) h4 (* h5
 h5 h5 h5) j2) (* 21 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 247 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 924 (* h1 h1) 
(* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2)) (* 1200 (* h1 h1) (* h3 h3 h3) h4 (* 
h5 h5 h5) h6 j2) (* 448 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6) (* 33 (* h1 
h1) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 391 (* h1 h1) (* h3 
h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1488 (* h1 h1) (* h3 h3 h3) h4 
(* h5 h5) (* h6 h6) (* j2 j2)) (* 2016 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* h6
 h6) j2) (* 832 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6)) (* 60 (* h1 h1) 
(* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2)) (* 348 (* h1 h1) (* h3 h3 h3) h4 
h5 (* h6 h6 h6) (* j2 j2)) (* 480 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) j2) 
(* 192 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6)) (* 16 (* h1 h1) (* h3 h3 h3) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 96 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6
 (* j2 j2)) (* 144 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6 j2) (* 64 (* h1 h1)
 (* h3 h3 h3) (* h5 h5 h5 h5) h6) (* 12 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2)) (* 144 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2)) (* 564 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 
816 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) j2) (* 384 (* h1 h1) (* h3 h3 
h3) (* h5 h5 h5) (* h6 h6)) (* 12 (* h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2)) (* 144 (* h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2)) (* 564 (* h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 816 (* 
h1 h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) j2) (* 384 (* h1 h1) (* h3 h3 h3) (* 
h5 h5) (* h6 h6 h6)) (* 16 (* h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2
)) (* 96 (* h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2)) (* 144 (* h1 h1) 
(* h3 h3 h3) h5 (* h6 h6 h6 h6) j2) (* 64 (* h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6 
h6)) (* 6 (* h1 h1) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 30 (* 
h1 h1) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2)) (* 24 (* h1 h1) (* h3 h3) 
(* h4 h4 h4 h4) (* h5 h5) j2) (* 12 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5
) (* j2 j2 j2)) (* 66 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) 
(* 72 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) j2) (* 26 (* h1 h1) (* h3 h3
) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 138 (* h1 h1) (* h3 h3) (* h4 h4 h4
) (* h5 h5) h6 (* j2 j2)) (* 144 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 
j2) (* 32 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6) (* 6 (* h1 h1) (* h3 h3
) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 30 (* h1 h1) (* h3 h3) (* h4 h4) 
(* h5 h5 h5 h5) (* j2 j2)) (* 24 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) 
j2) (* 40 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 233 (* 
h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 316 (* h1 h1) (* h3 h3)
 (* h4 h4) (* h5 h5 h5) h6 j2) (* 96 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) 
h6) (* 42 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 234
 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 288 (* h1 h1) 
(* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 96 (* h1 h1) (* h3 h3) (* h4 h4) 
(* h5 h5) (* h6 h6)) (* 14 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2
)) (* 78 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 96 (* h1 h1) 
(* h3 h3) h4 (* h5 h5 h5 h5) h6 j2) (* 32 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5)
 h6) (* 44 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 267 
(* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 408 (* h1 h1) (* h3 
h3) h4 (* h5 h5 h5) (* h6 h6) j2) (* 176 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* 
h6 h6)) (* 30 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 174
 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 240 (* h1 h1) (* h3
 h3) h4 (* h5 h5) (* h6 h6 h6) j2) (* 96 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 
h6 h6)) (* 8 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 48 
(* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 72 (* h1 h1) (* h3 
h3) (* h5 h5 h5 h5) (* h6 h6) j2) (* 32 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* 
h6 h6)) (* 16 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 100
 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 164 (* h1 h1) (* h3
 h3) (* h5 h5 h5) (* h6 h6 h6) j2) (* 80 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 
h6 h6)) (* 8 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 48 
(* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 72 (* h1 h1) (* h3 
h3) (* h5 h5) (* h6 h6 h6 h6) j2) (* 32 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 
h6 h6)) (* 3 (* h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 3 (* h1 h1)
 h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 13 (* h1 h1) h3 (* h4 h4 h4) (* 
h5 h5 h5) h6 (* j2 j2)) (* 7 (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) h6 j2) (* 10
 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 7 (* h1 h1) h3 (* h4 h4
) (* h5 h5 h5 h5) h6 j2) (* 21 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* 
j2 j2)) (* 22 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* 4 (* h1 h1) 
h3 (* h4 h4) (* h5 h5 h5) (* h6 h6)) (* 11 (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6
 h6) (* j2 j2)) (* 15 (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* 4 (* h1 
h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6)) (* 15 (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6
 h6) (* j2 j2)) (* 23 (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 8 (* h1 
h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6)) (* 4 (* h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6 
h6) (* j2 j2)) (* 8 (* h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* 4 (* h1 h1)
 h3 (* h5 h5 h5 h5) (* h6 h6 h6)) (* 4 (* h1 h1) h3 (* h5 h5 h5) (* h6 h6 h6 h6)
 (* j2 j2)) (* 8 (* h1 h1) h3 (* h5 h5 h5) (* h6 h6 h6 h6) j2) (* 4 (* h1 h1) h3
 (* h5 h5 h5) (* h6 h6 h6 h6)) (* 432 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 (* 
j2 j2 j2 j2)) (* 1296 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 
1440 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2)) (* 704 h1 (* h2 h2 h2 h2) 
(* h3 h3 h3) h4 h5 j2) (* 128 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5) (* 216 h1 
(* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 648 h1 (* h2 h2 h2 h2
) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 720 h1 (* h2 h2 h2 h2) (* h3 h3 h3) 
(* h5 h5) (* j2 j2)) (* 352 h1 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) j2) (* 64 
h1 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5)) (* 216 h1 (* h2 h2 h2 h2) (* h3 h3 h3
) h5 h6 (* j2 j2 j2 j2)) (* 648 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 
j2)) (* 720 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2)) (* 352 h1 (* h2 h2 
h2 h2) (* h3 h3 h3) h5 h6 j2) (* 64 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6) (* 
108 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 360 h1 (* h2 
h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 444 h1 (* h2 h2 h2 h2) (* h3 
h3) (* h4 h4) h5 (* j2 j2)) (* 240 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 j2)
 (* 48 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5) (* 144 h1 (* h2 h2 h2 h2) (* 
h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 480 h1 (* h2 h2 h2 h2) (* h3 h3) h4 (* 
h5 h5) (* j2 j2 j2)) (* 592 h1 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2))
 (* 320 h1 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) j2) (* 64 h1 (* h2 h2 h2 h2) 
(* h3 h3) h4 (* h5 h5)) (* 180 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2
 j2)) (* 600 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 740 h1 (* h2
 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2)) (* 400 h1 (* h2 h2 h2 h2) (* h3 h3) h4 
h5 h6 j2) (* 80 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6) (* 36 h1 (* h2 h2 h2 h2) 
(* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 120 h1 (* h2 h2 h2 h2) (* h3 h3) (* 
h5 h5 h5) (* j2 j2 j2)) (* 148 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 
j2)) (* 80 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) j2) (* 16 h1 (* h2 h2 h2 h2
) (* h3 h3) (* h5 h5 h5)) (* 108 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2
 j2 j2 j2)) (* 360 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 
444 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2)) (* 240 h1 (* h2 h2 h2 
h2) (* h3 h3) (* h5 h5) h6 j2) (* 48 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6) 
(* 72 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 240 h1 (* h2
 h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 296 h1 (* h2 h2 h2 h2) (* h3 
h3) h5 (* h6 h6) (* j2 j2)) (* 160 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) j2)
 (* 32 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6)) (* 18 h1 (* h2 h2 h2 h2) h3 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 66 h1 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5
 h5) (* j2 j2 j2)) (* 90 h1 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2)) 
(* 54 h1 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) j2) (* 12 h1 (* h2 h2 h2 h2) h3 
(* h4 h4) (* h5 h5)) (* 12 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2)
) (* 44 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2)) (* 60 h1 (* h2 h2 h2
 h2) h3 h4 (* h5 h5 h5) (* j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) 
j2) (* 8 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5)) (* 30 h1 (* h2 h2 h2 h2) h3 h4 
(* h5 h5) h6 (* j2 j2 j2 j2)) (* 110 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2
 j2 j2)) (* 150 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2)) (* 90 h1 (* h2 
h2 h2 h2) h3 h4 (* h5 h5) h6 j2) (* 20 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5) h6) 
(* 12 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 44 h1 (* h2 h2 
h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 60 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5)
 h6 (* j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 j2) (* 8 h1 (* h2 h2 
h2 h2) h3 (* h5 h5 h5) h6) (* 12 h1 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2
 j2 j2 j2)) (* 44 h1 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 60 
h1 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3
 (* h5 h5) (* h6 h6) j2) (* 8 h1 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6)) (* h1 
(* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 6 h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5)
 h6 (* j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 j2) (* h1 (* h2 h2
 h2 h2) (* h4 h4) (* h5 h5) h6) (* h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 
j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 6 h1 (* 
h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2)) (* 4 h1 (* h2 h2 h2 h2) h4 (* h5 h5 
h5) h6 j2) (* h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6) (* 2 h1 (* h2 h2 h2 h2) h4 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 8 h1 (* h2 h2 h2 h2) h4 (* h5 h5) (* h6 
h6) (* j2 j2 j2)) (* 12 h1 (* h2 h2 h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 
8 h1 (* h2 h2 h2 h2) h4 (* h5 h5) (* h6 h6) j2) (* 2 h1 (* h2 h2 h2 h2) h4 (* h5
 h5) (* h6 h6)) (* h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 4 h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 6 h1 (* h2 h2 h2
 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* 
h6 h6) j2) (* h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6)) (* h1 (* h2 h2 h2 h2) 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 
h6 h6) (* j2 j2 j2)) (* 6 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2)) 
(* 4 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) j2) (* h1 (* h2 h2 h2 h2) (* h5 
h5) (* h6 h6 h6)) (* 432 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) 
(* 2592 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 4032 h1 (* h2 h2 
h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2)) (* 2432 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 
h5 j2) (* 512 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5) (* 216 h1 (* h2 h2 h2) (* 
h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 1296 h1 (* h2 h2 h2) (* h3 h3 h3 h3) 
(* h5 h5) (* j2 j2 j2)) (* 2016 h1 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 
j2)) (* 1216 h1 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) j2) (* 256 h1 (* h2 h2 h2
) (* h3 h3 h3 h3) (* h5 h5)) (* 216 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 
j2 j2 j2)) (* 1296 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2)) (* 2016 
h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2)) (* 1216 h1 (* h2 h2 h2) (* h3 
h3 h3 h3) h5 h6 j2) (* 256 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6) (* 828 h1 (* 
h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 3228 h1 (* h2 h2 h2) (* 
h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 4496 h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 
h4) h5 (* j2 j2)) (* 2672 h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 j2) (* 576 
h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5) (* 864 h1 (* h2 h2 h2) (* h3 h3 h3) 
h4 (* h5 h5) (* j2 j2 j2 j2)) (* 3456 h1 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) 
(* j2 j2 j2)) (* 4896 h1 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 
2944 h1 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) j2) (* 640 h1 (* h2 h2 h2) (* h3 
h3 h3) h4 (* h5 h5)) (* 1584 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 
j2)) (* 6000 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 8192 h1 (* 
h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 4800 h1 (* h2 h2 h2) (* h3 h3 h3) 
h4 h5 h6 j2) (* 1024 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6) (* 72 h1 (* h2 h2 h2
) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 456 h1 (* h2 h2 h2) (* h3 h3 h3)
 (* h5 h5 h5) (* j2 j2 j2)) (* 800 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* 
j2 j2)) (* 544 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) j2) (* 128 h1 (* h2 h2 
h2) (* h3 h3 h3) (* h5 h5 h5)) (* 792 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2)) (* 3000 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2))
 (* 4096 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2)) (* 2400 h1 (* h2 
h2 h2) (* h3 h3 h3) (* h5 h5) h6 j2) (* 512 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 
h5) h6) (* 756 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 
2772 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 3696 h1 (* h2 h2
 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 2128 h1 (* h2 h2 h2) (* h3 h3 h3) 
h5 (* h6 h6) j2) (* 448 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6)) (* 114 h1 (* 
h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 520 h1 (* h2 h2 h2) (* 
h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 794 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4 
h4) h5 (* j2 j2)) (* 500 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 j2) (* 112 h1
 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5) (* 282 h1 (* h2 h2 h2) (* h3 h3) (* h4 
h4) (* h5 h5) (* j2 j2 j2 j2)) (* 1148 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5
 h5) (* j2 j2 j2)) (* 1702 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 
j2)) (* 1092 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) j2) (* 256 h1 (* h2 
h2 h2) (* h3 h3) (* h4 h4) (* h5 h5)) (* 342 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4)
 h5 h6 (* j2 j2 j2 j2)) (* 1524 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 
j2 j2)) (* 2298 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 1436 h1 
(* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 j2) (* 320 h1 (* h2 h2 h2) (* h3 h3) (* 
h4 h4) h5 h6) (* 144 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) 
(* 612 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 932 h1 (* h2 
h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 608 h1 (* h2 h2 h2) (* h3 h3) h4 
(* h5 h5 h5) j2) (* 144 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5)) (* 546 h1 (* 
h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 2176 h1 (* h2 h2 h2) (* 
h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 3170 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 
h5) h6 (* j2 j2)) (* 2004 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 j2) (* 464 
h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6) (* 342 h1 (* h2 h2 h2) (* h3 h3) h4 
h5 (* h6 h6) (* j2 j2 j2 j2)) (* 1488 h1 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2)) (* 2214 h1 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2)) (* 
1372 h1 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) j2) (* 304 h1 (* h2 h2 h2) (* h3 
h3) h4 h5 (* h6 h6)) (* 36 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2
)) (* 84 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2)) (* 64 h1 (* h2 h2 
h2) (* h3 h3) (* h5 h5 h5 h5) j2) (* 16 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5
)) (* 144 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 576 h1 
(* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 848 h1 (* h2 h2 h2) (* 
h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 544 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) 
h6 j2) (* 128 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6) (* 264 h1 (* h2 h2 h2) 
(* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 1028 h1 (* h2 h2 h2) (* h3 h3)
 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1468 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2)) (* 912 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) j2) 
(* 208 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6)) (* 114 h1 (* h2 h2 h2) (* 
h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 484 h1 (* h2 h2 h2) (* h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2)) (* 710 h1 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 
j2)) (* 436 h1 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) j2) (* 96 h1 (* h2 h2 h2) 
(* h3 h3) h5 (* h6 h6 h6)) (* 21 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2
 j2 j2 j2)) (* 93 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 147
 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2)) (* 99 h1 (* h2 h2 h2) h3 
(* h4 h4 h4) (* h5 h5) j2) (* 24 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5)) (* 
21 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 99 h1 (* h2 h2 
h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 163 h1 (* h2 h2 h2) h3 (* h4 h4) 
(* h5 h5 h5) (* j2 j2)) (* 113 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) j2) (* 
28 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5)) (* 67 h1 (* h2 h2 h2) h3 (* h4 h4)
 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 283 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 
(* j2 j2 j2)) (* 437 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 293
 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 j2) (* 72 h1 (* h2 h2 h2) h3 (* h4 h4
) (* h5 h5) h6) (* 12 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 32 
h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2)) (* 28 h1 (* h2 h2 h2) h3 h4 (* 
h5 h5 h5 h5) j2) (* 8 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5)) (* 46 h1 (* h2 h2 
h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 202 h1 (* h2 h2 h2) h3 h4 (* h5 h5
 h5) h6 (* j2 j2 j2)) (* 322 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2)) 
(* 222 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5) h6 j2) (* 56 h1 (* h2 h2 h2) h3 h4 (* 
h5 h5 h5) h6) (* 71 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 287 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 433 h1 (* h2 
h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 289 h1 (* h2 h2 h2) h3 h4 (* h5 
h5) (* h6 h6) j2) (* 72 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6)) (* 12 h1 (* 
h2 h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 32 h1 (* h2 h2 h2) h3 (* h5 h5 
h5 h5) h6 (* j2 j2)) (* 28 h1 (* h2 h2 h2) h3 (* h5 h5 h5 h5) h6 j2) (* 8 h1 (* 
h2 h2 h2) h3 (* h5 h5 h5 h5) h6) (* 25 h1 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6)
 (* j2 j2 j2 j2)) (* 103 h1 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2))
 (* 159 h1 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 109 h1 (* h2 h2 
h2) h3 (* h5 h5 h5) (* h6 h6) j2) (* 28 h1 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6
)) (* 25 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 97 h1 (* 
h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 143 h1 (* h2 h2 h2) h3 (* 
h5 h5) (* h6 h6 h6) (* j2 j2)) (* 95 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) 
j2) (* 24 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6)) (* h1 (* h2 h2 h2) (* h4 h4
 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 3 h1 (* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 
(* j2 j2)) (* 3 h1 (* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 j2) (* h1 (* h2 h2 h2)
 (* h4 h4 h4) (* h5 h5) h6) (* 2 h1 (* h2 h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2
 j2 j2)) (* 6 h1 (* h2 h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 6 h1 (* h2
 h2 h2) (* h4 h4) (* h5 h5 h5) h6 j2) (* 2 h1 (* h2 h2 h2) (* h4 h4) (* h5 h5 h5
) h6) (* 3 h1 (* h2 h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 9 h1 
(* h2 h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 9 h1 (* h2 h2 h2) (* h4
 h4) (* h5 h5) (* h6 h6) j2) (* 3 h1 (* h2 h2 h2) (* h4 h4) (* h5 h5) (* h6 h6))
 (* h1 (* h2 h2 h2) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 3 h1 (* h2 h2 h2) h4 
(* h5 h5 h5 h5) h6 (* j2 j2)) (* 3 h1 (* h2 h2 h2) h4 (* h5 h5 h5 h5) h6 j2) (* 
h1 (* h2 h2 h2) h4 (* h5 h5 h5 h5) h6) (* 4 h1 (* h2 h2 h2) h4 (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2)) (* 12 h1 (* h2 h2 h2) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) 
(* 12 h1 (* h2 h2 h2) h4 (* h5 h5 h5) (* h6 h6) j2) (* 4 h1 (* h2 h2 h2) h4 (* 
h5 h5 h5) (* h6 h6)) (* 3 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)
) (* 9 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 9 h1 (* h2 h2 h2)
 h4 (* h5 h5) (* h6 h6 h6) j2) (* 3 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6)) 
(* h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 3 h1 (* h2 h2 h2) 
(* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 
h6) j2) (* h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6)) (* 2 h1 (* h2 h2 h2) (* h5
 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 6 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6
) (* j2 j2)) (* 6 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6) j2) (* 2 h1 (* h2 h2
 h2) (* h5 h5 h5) (* h6 h6 h6)) (* h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* 
j2 j2 j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 3 h1 (* 
h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) j2) (* h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6
 h6)) (* 504 h1 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 3408 
h1 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 6752 h1 (* h2 h2) (* 
h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2)) (* 5056 h1 (* h2 h2) (* h3 h3 h3 h3) (* h4 
h4) h5 j2) (* 1280 h1 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5) (* 504 h1 (* h2 h2
) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 3408 h1 (* h2 h2) (* h3 h3 h3
 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 6752 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5
) (* j2 j2)) (* 5056 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) j2) (* 1280 h1 
(* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5)) (* 972 h1 (* h2 h2) (* h3 h3 h3 h3) h4 
h5 h6 (* j2 j2 j2 j2)) (* 6480 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2
)) (* 12528 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 9216 h1 (* h2 h2
) (* h3 h3 h3 h3) h4 h5 h6 j2) (* 2304 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6) 
(* 36 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 336 h1 (* h2
 h2) (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2)) (* 976 h1 (* h2 h2) (* h3 h3 h3 
h3) (* h5 h5 h5) (* j2 j2)) (* 896 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) j2)
 (* 256 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5)) (* 468 h1 (* h2 h2) (* h3 h3 
h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 3072 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 
h5) h6 (* j2 j2 j2)) (* 5776 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2)
) (* 4160 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 j2) (* 1024 h1 (* h2 h2) (* 
h3 h3 h3 h3) (* h5 h5) h6) (* 468 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* 
j2 j2 j2 j2)) (* 3072 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) 
(* 5776 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 4160 h1 (* h2 h2
) (* h3 h3 h3 h3) h5 (* h6 h6) j2) (* 1024 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6
 h6)) (* 156 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 1124 
h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 2456 h1 (* h2 h2) (* 
h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 1952 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 
h4) h5 j2) (* 512 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5) (* 366 h1 (* h2 h2)
 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 2554 h1 (* h2 h2) (* h3 h3
 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 5380 h1 (* h2 h2) (* h3 h3 h3) (* h4 
h4) (* h5 h5) (* j2 j2)) (* 4384 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5) 
j2) (* 1216 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5)) (* 468 h1 (* h2 h2) 
(* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 3336 h1 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) h5 h6 (* j2 j2 j2)) (* 7176 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 
(* j2 j2)) (* 5648 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 j2) (* 1472 h1 (* 
h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6) (* 180 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 
h5 h5) (* j2 j2 j2 j2)) (* 1284 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 
j2 j2)) (* 2792 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 2304 h1 
(* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) j2) (* 640 h1 (* h2 h2) (* h3 h3 h3) h4 
(* h5 h5 h5)) (* 714 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) 
(* 4934 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 10220 h1 (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 8192 h1 (* h2 h2) (* h3 h3 h3)
 h4 (* h5 h5) h6 j2) (* 2240 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6) (* 468 
h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 3300 h1 (* h2 h2) 
(* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2)) (* 6984 h1 (* h2 h2) (* h3 h3 h3) h4 
h5 (* h6 h6) (* j2 j2)) (* 5440 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) j2) 
(* 1408 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6)) (* 36 h1 (* h2 h2) (* h3 h3 
h3) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 
h5) (* j2 j2)) (* 208 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) j2) (* 64 h1 (* 
h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5)) (* 180 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 
h5) h6 (* j2 j2 j2 j2)) (* 1248 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 
j2 j2)) (* 2600 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 2096 h1 
(* h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 j2) (* 576 h1 (* h2 h2) (* h3 h3 h3) (* 
h5 h5 h5) h6) (* 348 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2)) (* 2380 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 4840
 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 3808 h1 (* h2 h2) 
(* h3 h3 h3) (* h5 h5) (* h6 h6) j2) (* 1024 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5)
 (* h6 h6)) (* 156 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) 
(* 1088 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 2264 h1 (* h2
 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 1744 h1 (* h2 h2) (* h3 h3 h3) 
h5 (* h6 h6 h6) j2) (* 448 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6)) (* 78 h1 
(* h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 178 h1 (* h2 h2) (* h3 
h3) (* h4 h4 h4 h4) h5 (* j2 j2)) (* 132 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4 h4) 
h5 j2) (* 32 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5) (* 45 h1 (* h2 h2) (* h3
 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 414 h1 (* h2 h2) (* h3 h3) (* h4
 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 965 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5
 h5) (* j2 j2)) (* 836 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 240 
h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5)) (* 312 h1 (* h2 h2) (* h3 h3) (* 
h4 h4 h4) h5 h6 (* j2 j2 j2)) (* 712 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 
(* j2 j2)) (* 528 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 j2) (* 128 h1 (* h2 
h2) (* h3 h3) (* h4 h4 h4) h5 h6) (* 45 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2)) (* 426 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2)) (* 997 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 
876 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 256 h1 (* h2 h2) (* h3 
h3) (* h4 h4) (* h5 h5 h5)) (* 141 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6
 (* j2 j2 j2 j2)) (* 1254 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2
 j2)) (* 2821 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 2396 
h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 j2) (* 688 h1 (* h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5) h6) (* 468 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* 
j2 j2 j2)) (* 1068 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 
792 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6) j2) (* 192 h1 (* h2 h2) (* h3 
h3) (* h4 h4) h5 (* h6 h6)) (* 96 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* 
j2 j2 j2)) (* 244 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 216 h1
 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) j2) (* 64 h1 (* h2 h2) (* h3 h3) h4 (* 
h5 h5 h5 h5)) (* 96 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 864 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 1920 h1 (* h2 
h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2)) (* 1640 h1 (* h2 h2) (* h3 h3) h4 
(* h5 h5 h5) h6 j2) (* 480 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6) (* 147 h1 
(* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 1266 h1 (* h2 h2)
 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 2747 h1 (* h2 h2) (* h3 h3) 
h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 2284 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) 
(* h6 h6) j2) (* 656 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6)) (* 312 h1 
(* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2)) (* 712 h1 (* h2 h2) (* h3 
h3) h4 h5 (* h6 h6 h6) (* j2 j2)) (* 528 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 
h6) j2) (* 128 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6)) (* 96 h1 (* h2 h2) (* 
h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 244 h1 (* h2 h2) (* h3 h3) (* h5 h5 
h5 h5) h6 (* j2 j2)) (* 216 h1 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 j2) (* 64 
h1 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6) (* 51 h1 (* h2 h2) (* h3 h3) (* h5 h5
 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 438 h1 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6
 h6) (* j2 j2 j2)) (* 923 h1 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2
)) (* 764 h1 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) j2) (* 224 h1 (* h2 h2) 
(* h3 h3) (* h5 h5 h5) (* h6 h6)) (* 51 h1 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 
h6 h6) (* j2 j2 j2 j2)) (* 426 h1 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2)) (* 891 h1 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 
724 h1 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) j2) (* 208 h1 (* h2 h2) (* h3 
h3) (* h5 h5) (* h6 h6 h6)) (* 78 h1 (* h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* 
j2 j2 j2)) (* 178 h1 (* h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2)) (* 132 h1
 (* h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) j2) (* 32 h1 (* h2 h2) (* h3 h3) h5 (* 
h6 h6 h6 h6)) (* 15 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 
32 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2)) (* 21 h1 (* h2 h2) h3 
(* h4 h4 h4 h4) (* h5 h5) j2) (* 4 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5)) 
(* 30 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 88 h1 (* h2 h2)
 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 82 h1 (* h2 h2) h3 (* h4 h4 h4) (* 
h5 h5 h5) j2) (* 24 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5)) (* 63 h1 (* h2 h2
) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 134 h1 (* h2 h2) h3 (* h4 h4 h4)
 (* h5 h5) h6 (* j2 j2)) (* 91 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 j2) (* 
20 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6) (* 15 h1 (* h2 h2) h3 (* h4 h4) 
(* h5 h5 h5 h5) (* j2 j2 j2)) (* 44 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) 
(* j2 j2)) (* 41 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) j2) (* 12 h1 (* h2 h2
) h3 (* h4 h4) (* h5 h5 h5 h5)) (* 96 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) h6 
(* j2 j2 j2)) (* 270 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 250
 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) h6 j2) (* 76 h1 (* h2 h2) h3 (* h4 h4) 
(* h5 h5 h5) h6) (* 99 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2
)) (* 210 h1 (* h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 147 h1 (* 
h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 36 h1 (* h2 h2) h3 (* h4 h4) (* 
h5 h5) (* h6 h6)) (* 33 h1 (* h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 
94 h1 (* h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 89 h1 (* h2 h2) h3 h4 (* 
h5 h5 h5 h5) h6 j2) (* 28 h1 (* h2 h2) h3 h4 (* h5 h5 h5 h5) h6) (* 102 h1 (* h2
 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 276 h1 (* h2 h2) h3 h4 (* h5 
h5 h5) (* h6 h6) (* j2 j2)) (* 254 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) j2)
 (* 80 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6)) (* 69 h1 (* h2 h2) h3 h4 (* h5
 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 146 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6)
 (* j2 j2)) (* 105 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) j2) (* 28 h1 (* h2 
h2) h3 h4 (* h5 h5) (* h6 h6 h6)) (* 18 h1 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6
) (* j2 j2 j2)) (* 50 h1 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 48
 h1 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) j2) (* 16 h1 (* h2 h2) h3 (* h5 h5 h5
 h5) (* h6 h6)) (* 36 h1 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) 
(* 94 h1 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 86 h1 (* h2 h2) h3
 (* h5 h5 h5) (* h6 h6 h6) j2) (* 28 h1 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6)) 
(* 18 h1 (* h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 38 h1 (* h2 h2)
 h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 28 h1 (* h2 h2) h3 (* h5 h5) (* h6 
h6 h6 h6) j2) (* 8 h1 (* h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6)) (* h1 (* h2 h2) 
(* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 2 h1 (* h2 h2) (* h4 h4 h4) (* h5 h5 
h5) h6 j2) (* h1 (* h2 h2) (* h4 h4 h4) (* h5 h5 h5) h6) (* h1 (* h2 h2) (* h4 
h4) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 2 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5 h5) h6
 j2) (* h1 (* h2 h2) (* h4 h4) (* h5 h5 h5 h5) h6) (* 3 h1 (* h2 h2) (* h4 h4) 
(* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 6 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5) (* h6
 h6) j2) (* 3 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5) (* h6 h6)) (* 2 h1 (* h2 h2) 
h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 4 h1 (* h2 h2) h4 (* h5 h5 h5 h5) (* 
h6 h6) j2) (* 2 h1 (* h2 h2) h4 (* h5 h5 h5 h5) (* h6 h6)) (* 3 h1 (* h2 h2) h4 
(* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 6 h1 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6 
h6) j2) (* 3 h1 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6 h6)) (* h1 (* h2 h2) (* h5 h5
 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 2 h1 (* h2 h2) (* h5 h5 h5 h5) (* h6 h6 h6) 
j2) (* h1 (* h2 h2) (* h5 h5 h5 h5) (* h6 h6 h6)) (* h1 (* h2 h2) (* h5 h5 h5) 
(* h6 h6 h6 h6) (* j2 j2)) (* 2 h1 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6 h6) j2) 
(* h1 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6 h6)) (* 42 h1 h2 (* h3 h3 h3 h3) (* h4 
h4 h4) h5 (* j2 j2 j2 j2)) (* 388 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2
 j2)) (* 1104 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 960 h1 h2 (* 
h3 h3 h3 h3) (* h4 h4 h4) h5 j2) (* 256 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5) 
(* 102 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 1004 h1 h2 
(* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 3216 h1 h2 (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2)) (* 3648 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)
 j2) (* 1280 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)) (* 126 h1 h2 (* h3 h3 h3
 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 1164 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5
 h6 (* j2 j2 j2)) (* 3312 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 
2880 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 j2) (* 768 h1 h2 (* h3 h3 h3 h3) (* 
h4 h4) h5 h6) (* 48 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 
464 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 1440 h1 h2 (* h3 h3 
h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 1536 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) 
j2) (* 512 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5)) (* 198 h1 h2 (* h3 h3 h3 h3) 
h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1932 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 
(* j2 j2 j2)) (* 6096 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 6720 
h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 j2) (* 2304 h1 h2 (* h3 h3 h3 h3) h4 (* h5
 h5) h6) (* 126 h1 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 1164 
h1 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2)) (* 3312 h1 h2 (* h3 h3 h3 h3
) h4 h5 (* h6 h6) (* j2 j2)) (* 2880 h1 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) j2) 
(* 768 h1 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6)) (* 48 h1 h2 (* h3 h3 h3 h3) (* h5 
h5 h5) h6 (* j2 j2 j2 j2)) (* 464 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2
 j2)) (* 1440 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 1536 h1 h2 (* 
h3 h3 h3 h3) (* h5 h5 h5) h6 j2) (* 512 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6) 
(* 96 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 928 h1 h2 
(* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 2880 h1 h2 (* h3 h3 h3 h3) 
(* h5 h5) (* h6 h6) (* j2 j2)) (* 3072 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6)
 j2) (* 1024 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6)) (* 42 h1 h2 (* h3 h3 h3 
h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 388 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6
) (* j2 j2 j2)) (* 1104 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 960 
h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) j2) (* 256 h1 h2 (* h3 h3 h3 h3) h5 (* h6 
h6 h6)) (* 42 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 220 h1 h2 
(* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2)) (* 224 h1 h2 (* h3 h3 h3) (* h4 h4 h4
 h4) h5 j2) (* 64 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5) (* 27 h1 h2 (* h3 h3 h3
) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 321 h1 h2 (* h3 h3 h3) (* h4 h4 h4)
 (* h5 h5) (* j2 j2 j2)) (* 1184 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2
 j2)) (* 1456 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 512 h1 h2 (* h3 
h3 h3) (* h4 h4 h4) (* h5 h5)) (* 168 h1 h2 (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* 
j2 j2 j2)) (* 880 h1 h2 (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2)) (* 896 h1 h2 
(* h3 h3 h3) (* h4 h4 h4) h5 h6 j2) (* 256 h1 h2 (* h3 h3 h3) (* h4 h4 h4) h5 h6
) (* 27 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 327 h1 h2 
(* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 1236 h1 h2 (* h3 h3 h3) (* 
h4 h4) (* h5 h5 h5) (* j2 j2)) (* 1584 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5)
 j2) (* 576 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5)) (* 85 h1 h2 (* h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 991 h1 h2 (* h3 h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2)) (* 3584 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 
j2)) (* 4304 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 j2) (* 1536 h1 h2 (* h3 
h3 h3) (* h4 h4) (* h5 h5) h6) (* 252 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) 
(* j2 j2 j2)) (* 1320 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 
1344 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) j2) (* 384 h1 h2 (* h3 h3 h3) (* 
h4 h4) h5 (* h6 h6)) (* 48 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) 
(* 272 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 352 h1 h2 (* h3 h3 h3
) h4 (* h5 h5 h5 h5) j2) (* 128 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5)) (* 58 h1 
h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 682 h1 h2 (* h3 h3 h3) h4
 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 2504 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* 
j2 j2)) (* 3104 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 j2) (* 1152 h1 h2 (* h3 h3
 h3) h4 (* h5 h5 h5) h6) (* 89 h1 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2)) (* 1019 h1 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
3616 h1 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 4240 h1 h2 (* h3 h3
 h3) h4 (* h5 h5) (* h6 h6) j2) (* 1536 h1 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6
)) (* 168 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2)) (* 880 h1 h2 (* h3
 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2)) (* 896 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 
h6) j2) (* 256 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 h6)) (* 48 h1 h2 (* h3 h3 h3) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 272 h1 h2 (* h3 h3 h3) (* h5 h5 h5 h5) h6 
(* j2 j2)) (* 352 h1 h2 (* h3 h3 h3) (* h5 h5 h5 h5) h6 j2) (* 128 h1 h2 (* h3 
h3 h3) (* h5 h5 h5 h5) h6) (* 31 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2
 j2 j2 j2)) (* 355 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
1268 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 1520 h1 h2 (* h3 h3
 h3) (* h5 h5 h5) (* h6 h6) j2) (* 576 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6)
) (* 31 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 349 h1 h2 
(* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 1216 h1 h2 (* h3 h3 h3) (* 
h5 h5) (* h6 h6 h6) (* j2 j2)) (* 1392 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6)
 j2) (* 512 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6)) (* 42 h1 h2 (* h3 h3 h3) 
h5 (* h6 h6 h6 h6) (* j2 j2 j2)) (* 220 h1 h2 (* h3 h3 h3) h5 (* h6 h6 h6 h6) 
(* j2 j2)) (* 224 h1 h2 (* h3 h3 h3) h5 (* h6 h6 h6 h6) j2) (* 64 h1 h2 (* h3 h3
 h3) h5 (* h6 h6 h6 h6)) (* 18 h1 h2 (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 
j2 j2)) (* 98 h1 h2 (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2)) (* 112 h1 h2 
(* h3 h3) (* h4 h4 h4 h4) (* h5 h5) j2) (* 32 h1 h2 (* h3 h3) (* h4 h4 h4 h4) 
(* h5 h5)) (* 36 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 214 
h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 308 h1 h2 (* h3 h3) (* 
h4 h4 h4) (* h5 h5 h5) j2) (* 112 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5)) (* 
75 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 407 h1 h2 (* h3 h3
) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 464 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5
 h5) h6 j2) (* 144 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6) (* 18 h1 h2 (* h3 
h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 104 h1 h2 (* h3 h3) (* h4 h4) (* 
h5 h5 h5 h5) (* j2 j2)) (* 140 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) j2) (* 
48 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5)) (* 114 h1 h2 (* h3 h3) (* h4 h4) 
(* h5 h5 h5) h6 (* j2 j2 j2)) (* 666 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 
(* j2 j2)) (* 928 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 j2) (* 352 h1 h2 (* 
h3 h3) (* h4 h4) (* h5 h5 h5) h6) (* 117 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* 
h6 h6) (* j2 j2 j2)) (* 633 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 
j2)) (* 720 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 240 h1 h2 (* h3
 h3) (* h4 h4) (* h5 h5) (* h6 h6)) (* 39 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 
(* j2 j2 j2)) (* 223 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 296 h1 
h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 j2) (* 112 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5)
 h6) (* 120 h1 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 690 h1 h2
 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 932 h1 h2 (* h3 h3) h4 (* h5 
h5 h5) (* h6 h6) j2) (* 368 h1 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6)) (* 81 h1 
h2 (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 437 h1 h2 (* h3 h3) h4 
(* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 496 h1 h2 (* h3 h3) h4 (* h5 h5) (* h6 h6 
h6) j2) (* 176 h1 h2 (* h3 h3) h4 (* h5 h5) (* h6 h6 h6)) (* 21 h1 h2 (* h3 h3) 
(* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 119 h1 h2 (* h3 h3) (* h5 h5 h5 h5) 
(* h6 h6) (* j2 j2)) (* 156 h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) j2) (* 64 
h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6)) (* 42 h1 h2 (* h3 h3) (* h5 h5 h5) 
(* h6 h6 h6) (* j2 j2 j2)) (* 238 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* 
j2 j2)) (* 312 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) j2) (* 128 h1 h2 (* h3 
h3) (* h5 h5 h5) (* h6 h6 h6)) (* 21 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) 
(* j2 j2 j2)) (* 113 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 128
 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) j2) (* 48 h1 h2 (* h3 h3) (* h5 h5) 
(* h6 h6 h6 h6)) (* 9 h1 h2 h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 7 h1 
h2 h3 (* h4 h4 h4 h4) (* h5 h5 h5) j2) (* 9 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5 h5
) (* j2 j2)) (* 7 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5 h5) j2) (* 38 h1 h2 h3 (* h4
 h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 38 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) h6 
j2) (* 8 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) h6) (* 29 h1 h2 h3 (* h4 h4) (* h5 
h5 h5 h5) h6 (* j2 j2)) (* 31 h1 h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6 j2) (* 8 h1 
h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6) (* 60 h1 h2 h3 (* h4 h4) (* h5 h5 h5) (* h6 
h6) (* j2 j2)) (* 72 h1 h2 h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* 24 h1 h2 
h3 (* h4 h4) (* h5 h5 h5) (* h6 h6)) (* 31 h1 h2 h3 h4 (* h5 h5 h5 h5) (* h6 h6)
 (* j2 j2)) (* 41 h1 h2 h3 h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* 16 h1 h2 h3 h4 
(* h5 h5 h5 h5) (* h6 h6)) (* 42 h1 h2 h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)
) (* 58 h1 h2 h3 h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 24 h1 h2 h3 h4 (* h5 h5 h5)
 (* h6 h6 h6)) (* 11 h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 17 h1 
h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* 8 h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6
)) (* 11 h1 h2 h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 17 h1 h2 h3 (* h5 
h5 h5) (* h6 h6 h6 h6) j2) (* 8 h1 h2 h3 (* h5 h5 h5) (* h6 h6 h6 h6)) (* 3 h1 
(* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 36 h1 (* h3 h3 h3 h3)
 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 144 h1 (* h3 h3 h3 h3) (* h4 h4 h4) (* 
h5 h5) (* j2 j2)) (* 192 h1 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 3 h1 
(* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 36 h1 (* h3 h3 h3 h3)
 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 144 h1 (* h3 h3 h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2)) (* 192 h1 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 10 h1 
(* h3 h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 124 h1 (* h3 h3 h3 h3
) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 528 h1 (* h3 h3 h3 h3) (* h4 h4) (* h5
 h5) h6 (* j2 j2)) (* 832 h1 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) h6 j2) (* 256 
h1 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) h6) (* 7 h1 (* h3 h3 h3 h3) h4 (* h5 h5 
h5) h6 (* j2 j2 j2 j2)) (* 88 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)
) (* 384 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2)) (* 640 h1 (* h3 h3 h3 
h3) h4 (* h5 h5 h5) h6 j2) (* 256 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6) (* 11 
h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 140 h1 (* h3 h3 h3
 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 624 h1 (* h3 h3 h3 h3) h4 (* h5 h5)
 (* h6 h6) (* j2 j2)) (* 1088 h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 
512 h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6)) (* 4 h1 (* h3 h3 h3 h3) (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2)) (* 52 h1 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2)) (* 240 h1 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 448
 h1 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6) j2) (* 256 h1 (* h3 h3 h3 h3) (* h5 
h5 h5) (* h6 h6)) (* 4 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)
) (* 52 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 240 h1 (* h3 
h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 448 h1 (* h3 h3 h3 h3) (* h5 h5) 
(* h6 h6 h6) j2) (* 256 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6)) (* 3 h1 (* h3
 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 24 h1 (* h3 h3 h3) (* h4 h4 
h4 h4) (* h5 h5) (* j2 j2)) (* 48 h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) j2) 
(* 6 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 48 h1 (* h3 h3 
h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 96 h1 (* h3 h3 h3) (* h4 h4 h4) (* 
h5 h5 h5) j2) (* 13 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 
108 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 240 h1 (* h3 h3 h3) 
(* h4 h4 h4) (* h5 h5) h6 j2) (* 64 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6) 
(* 3 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 24 h1 (* h3 h3 
h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 48 h1 (* h3 h3 h3) (* h4 h4) (* h5 
h5 h5 h5) j2) (* 20 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 
168 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 384 h1 (* h3 h3 h3) 
(* h4 h4) (* h5 h5 h5) h6 j2) (* 128 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6) 
(* 21 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 180 h1 (* 
h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 432 h1 (* h3 h3 h3) (* h4 
h4) (* h5 h5) (* h6 h6) j2) (* 192 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6)
) (* 7 h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 60 h1 (* h3 h3 h3)
 h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 144 h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 
j2) (* 64 h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6) (* 22 h1 (* h3 h3 h3) h4 (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2)) (* 192 h1 (* h3 h3 h3) h4 (* h5 h5 h5) (* h6 h6) 
(* j2 j2)) (* 480 h1 (* h3 h3 h3) h4 (* h5 h5 h5) (* h6 h6) j2) (* 256 h1 (* h3 
h3 h3) h4 (* h5 h5 h5) (* h6 h6)) (* 15 h1 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6 h6
) (* j2 j2 j2)) (* 132 h1 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 
336 h1 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6 h6) j2) (* 192 h1 (* h3 h3 h3) h4 (* 
h5 h5) (* h6 h6 h6)) (* 4 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)
) (* 36 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 96 h1 (* h3 h3 
h3) (* h5 h5 h5 h5) (* h6 h6) j2) (* 64 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6
)) (* 8 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 72 h1 (* h3 
h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 192 h1 (* h3 h3 h3) (* h5 h5 h5) 
(* h6 h6 h6) j2) (* 128 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6)) (* 4 h1 (* h3
 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 36 h1 (* h3 h3 h3) (* h5 h5) 
(* h6 h6 h6 h6) (* j2 j2)) (* 96 h1 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) j2) 
(* 64 h1 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6)) (* 3 h1 (* h3 h3) (* h4 h4 h4 
h4) (* h5 h5 h5) (* j2 j2)) (* 12 h1 (* h3 h3) (* h4 h4 h4 h4) (* h5 h5 h5) j2) 
(* 3 h1 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 12 h1 (* h3 h3) (* 
h4 h4 h4) (* h5 h5 h5 h5) j2) (* 13 h1 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 
(* j2 j2)) (* 56 h1 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 j2) (* 16 h1 (* h3 h3
) (* h4 h4 h4) (* h5 h5 h5) h6) (* 10 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 
(* j2 j2)) (* 44 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 j2) (* 16 h1 (* h3 h3
) (* h4 h4) (* h5 h5 h5 h5) h6) (* 21 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 
h6) (* j2 j2)) (* 96 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* 48 h1 
(* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6)) (* 11 h1 (* h3 h3) h4 (* h5 h5 h5 h5
) (* h6 h6) (* j2 j2)) (* 52 h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* 32
 h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6)) (* 15 h1 (* h3 h3) h4 (* h5 h5 h5) 
(* h6 h6 h6) (* j2 j2)) (* 72 h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 
48 h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6)) (* 4 h1 (* h3 h3) (* h5 h5 h5 h5) 
(* h6 h6 h6) (* j2 j2)) (* 20 h1 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* 
16 h1 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6)) (* 4 h1 (* h3 h3) (* h5 h5 h5) (* 
h6 h6 h6 h6) (* j2 j2)) (* 20 h1 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6) j2) (* 
16 h1 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6)) (* 216 (* h2 h2 h2 h2) (* h3 h3 h3
) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 648 (* h2 h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5
 (* j2 j2 j2)) (* 720 (* h2 h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2)) (* 
352 (* h2 h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 j2) (* 64 (* h2 h2 h2 h2) (* h3 h3
 h3) (* h4 h4) h5) (* 216 (* h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 
j2)) (* 648 (* h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 720 (* h2
 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 352 (* h2 h2 h2 h2) (* h3 h3 
h3) h4 (* h5 h5) j2) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5)) (* 432 (* 
h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 1296 (* h2 h2 h2 h2) (* 
h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 1440 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 
(* j2 j2)) (* 704 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 j2) (* 128 (* h2 h2 h2 
h2) (* h3 h3 h3) h4 h5 h6) (* 216 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* 
j2 j2 j2 j2)) (* 648 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 
720 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2)) (* 352 (* h2 h2 h2 h2) 
(* h3 h3 h3) (* h5 h5) h6 j2) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6) 
(* 216 (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 648 (* h2 
h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 720 (* h2 h2 h2 h2) (* h3 
h3 h3) h5 (* h6 h6) (* j2 j2)) (* 352 (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) 
j2) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6)) (* 36 (* h2 h2 h2 h2) (* h3
 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 
h4) h5 (* j2 j2 j2)) (* 148 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2))
 (* 80 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 j2) (* 16 (* h2 h2 h2 h2) (* h3
 h3) (* h4 h4 h4) h5) (* 72 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 
j2 j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) 
(* 296 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2)) (* 160 (* h2 h2 
h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) j2) (* 32 (* h2 h2 h2 h2) (* h3 h3) (* h4 
h4) (* h5 h5)) (* 108 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2))
 (* 360 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2)) (* 444 (* h2 h2 
h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3) (* 
h4 h4) h5 h6 j2) (* 48 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6) (* 36 (* h2 h2
 h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 
h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 148 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5 
h5) (* j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) j2) (* 16 (* h2 
h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5)) (* 144 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 
h5) h6 (* j2 j2 j2 j2)) (* 480 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 
j2 j2)) (* 592 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 320 (* h2
 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 j2) (* 64 (* h2 h2 h2 h2) (* h3 h3) h4 (* 
h5 h5) h6) (* 108 (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 
360 (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2)) (* 444 (* h2 h2 h2 
h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3) h4 h5 
(* h6 h6) j2) (* 48 (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6)) (* 36 (* h2 h2 h2
 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 h3)
 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 148 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 
(* j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 j2) (* 16 (* h2 h2 h2
 h2) (* h3 h3) (* h5 h5 h5) h6) (* 72 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 
j2 j2)) (* 296 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 160 
(* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) j2) (* 32 (* h2 h2 h2 h2) (* h3 h3
) (* h5 h5) (* h6 h6)) (* 36 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 
j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 148 
(* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3
 h3) h5 (* h6 h6 h6) j2) (* 16 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6)) (* 6 
(* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 22 (* h2 h2 h2 h2)
 h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 30 (* h2 h2 h2 h2) h3 (* h4 h4 h4) 
(* h5 h5) (* j2 j2)) (* 18 (* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) j2) (* 4 
(* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5)) (* 6 (* h2 h2 h2 h2) h3 (* h4 h4) (* 
h5 h5 h5) (* j2 j2 j2 j2)) (* 22 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2
 j2 j2)) (* 30 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 18 (* h2 
h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) j2) (* 4 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 
h5 h5)) (* 18 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 66 
(* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 90 (* h2 h2 h2 h2) h3
 (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 54 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) 
h6 j2) (* 12 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6) (* 12 (* h2 h2 h2 h2) h3
 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 44 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) h6
 (* j2 j2 j2)) (* 60 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2)) (* 36 (* 
h2 h2 h2 h2) h3 h4 (* h5 h5 h5) h6 j2) (* 8 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) 
h6) (* 18 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 66 (* h2
 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 90 (* h2 h2 h2 h2) h3 h4 
(* h5 h5) (* h6 h6) (* j2 j2)) (* 54 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) 
j2) (* 12 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6)) (* 6 (* h2 h2 h2 h2) h3 (* 
h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 22 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6
 h6) (* j2 j2 j2)) (* 30 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2)) 
(* 18 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) j2) (* 4 (* h2 h2 h2 h2) h3 (* 
h5 h5 h5) (* h6 h6)) (* 6 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2)) (* 22 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 30 (* h2 
h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 18 (* h2 h2 h2 h2) h3 (* h5 h5
) (* h6 h6 h6) j2) (* 4 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6)) (* 216 (* h2 
h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 
h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 
h4) h5 (* j2 j2)) (* 1216 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 j2) (* 256 
(* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5) (* 216 (* h2 h2 h2) (* h3 h3 h3 h3) 
h4 (* h5 h5) (* j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) 
(* j2 j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2)) (* 
1216 (* h2 h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) j2) (* 256 (* h2 h2 h2) (* h3 h3 
h3 h3) h4 (* h5 h5)) (* 432 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2
)) (* 2592 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 4032 (* h2 h2 
h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2)) (* 2432 (* h2 h2 h2) (* h3 h3 h3 h3) h4 
h5 h6 j2) (* 512 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6) (* 216 (* h2 h2 h2) (* 
h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) 
(* h5 h5) h6 (* j2 j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* 
j2 j2)) (* 1216 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 j2) (* 256 (* h2 h2 h2
) (* h3 h3 h3 h3) (* h5 h5) h6) (* 216 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6)
 (* j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)
) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2)) (* 1216 (* h2 h2 
h2) (* h3 h3 h3 h3) h5 (* h6 h6) j2) (* 256 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* 
h6 h6)) (* 72 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 456 
(* h2 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 800 (* h2 h2 h2) (* 
h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2)) (* 544 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4 
h4) h5 j2) (* 128 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5) (* 144 (* h2 h2 h2)
 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 912 (* h2 h2 h2) (* h3 h3 
h3) (* h4 h4) (* h5 h5) (* j2 j2 j2)) (* 1600 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4
) (* h5 h5) (* j2 j2)) (* 1088 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5) j2)
 (* 256 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5)) (* 216 (* h2 h2 h2) (* h3
 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 1368 (* h2 h2 h2) (* h3 h3 h3) (* h4
 h4) h5 h6 (* j2 j2 j2)) (* 2400 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2
 j2)) (* 1632 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 j2) (* 384 (* h2 h2 h2) 
(* h3 h3 h3) (* h4 h4) h5 h6) (* 72 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) 
(* j2 j2 j2 j2)) (* 456 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) 
(* 800 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 544 (* h2 h2 h2) 
(* h3 h3 h3) h4 (* h5 h5 h5) j2) (* 128 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5
)) (* 288 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1824 (* 
h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 3200 (* h2 h2 h2) (* h3 
h3 h3) h4 (* h5 h5) h6 (* j2 j2)) (* 2176 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5)
 h6 j2) (* 512 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6) (* 216 (* h2 h2 h2) 
(* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 1368 (* h2 h2 h2) (* h3 h3 h3) 
h4 h5 (* h6 h6) (* j2 j2 j2)) (* 2400 (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) 
(* j2 j2)) (* 1632 (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) j2) (* 384 (* h2 h2
 h2) (* h3 h3 h3) h4 h5 (* h6 h6)) (* 72 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2)) (* 456 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2
)) (* 800 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 544 (* h2 h2 
h2) (* h3 h3 h3) (* h5 h5 h5) h6 j2) (* 128 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 
h5) h6) (* 144 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 912 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1600 (* h2
 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 1088 (* h2 h2 h2) (* h3 
h3 h3) (* h5 h5) (* h6 h6) j2) (* 256 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 
h6)) (* 72 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 456 (* 
h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 800 (* h2 h2 h2) (* h3 
h3 h3) h5 (* h6 h6 h6) (* j2 j2)) (* 544 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6 
h6) j2) (* 128 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6)) (* 36 (* h2 h2 h2) (* 
h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 84 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4
 h4) h5 (* j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 j2) (* 16 (* 
h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5) (* 18 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4
) (* h5 h5) (* j2 j2 j2 j2)) (* 156 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5
) (* j2 j2 j2)) (* 318 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2)) 
(* 244 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 64 (* h2 h2 h2) (* 
h3 h3) (* h4 h4 h4) (* h5 h5)) (* 144 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 
(* j2 j2 j2)) (* 336 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2)) (* 256
 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 j2) (* 64 (* h2 h2 h2) (* h3 h3) (* 
h4 h4 h4) h5 h6) (* 18 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2
 j2)) (* 156 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 318 
(* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 244 (* h2 h2 h2) (* 
h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 64 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 
h5 h5)) (* 54 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 
468 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 954 (* h2 h2 
h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 732 (* h2 h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5) h6 j2) (* 192 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6)
 (* 216 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2)) (* 504 (* h2
 h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 384 (* h2 h2 h2) (* h3 h3
) (* h4 h4) h5 (* h6 h6) j2) (* 96 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* h6 h6)
) (* 36 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 84 (* h2 h2 
h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) h4 (* 
h5 h5 h5 h5) j2) (* 16 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5)) (* 36 (* h2 h2
 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 312 (* h2 h2 h2) (* h3 h3)
 h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 636 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) 
h6 (* j2 j2)) (* 488 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 j2) (* 128 (* h2 
h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6) (* 54 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2)) (* 468 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) 
(* j2 j2 j2)) (* 954 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2)) 
(* 732 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 192 (* h2 h2 h2) (* 
h3 h3) h4 (* h5 h5) (* h6 h6)) (* 144 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) 
(* j2 j2 j2)) (* 336 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2)) (* 256
 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) j2) (* 64 (* h2 h2 h2) (* h3 h3) h4 
h5 (* h6 h6 h6)) (* 36 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2)) 
(* 84 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 64 (* h2 h2 h2) 
(* h3 h3) (* h5 h5 h5 h5) h6 j2) (* 16 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6
) (* 18 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 156 
(* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 318 (* h2 h2 h2) 
(* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 244 (* h2 h2 h2) (* h3 h3) (* h5 
h5 h5) (* h6 h6) j2) (* 64 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6)) (* 18 
(* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 156 (* h2 h2 
h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 318 (* h2 h2 h2) (* h3 h3)
 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 244 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6
 h6 h6) j2) (* 64 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6)) (* 36 (* h2 h2 
h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2)) (* 84 (* h2 h2 h2) (* h3 h3) h5 
(* h6 h6 h6 h6) (* j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) j2) 
(* 16 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6)) (* 6 (* h2 h2 h2) h3 (* h4 h4 
h4 h4) (* h5 h5) (* j2 j2 j2)) (* 16 (* h2 h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) 
(* j2 j2)) (* 14 (* h2 h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) j2) (* 4 (* h2 h2 h2)
 h3 (* h4 h4 h4 h4) (* h5 h5)) (* 12 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) 
(* j2 j2 j2)) (* 32 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 28 
(* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) j2) (* 8 (* h2 h2 h2) h3 (* h4 h4 h4) 
(* h5 h5 h5)) (* 24 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 
64 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 56 (* h2 h2 h2) h3 
(* h4 h4 h4) (* h5 h5) h6 j2) (* 16 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6) 
(* 6 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 16 (* h2 h2 h2) 
h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 14 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5
 h5 h5) j2) (* 4 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5)) (* 36 (* h2 h2 h2) 
h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 96 (* h2 h2 h2) h3 (* h4 h4) (* h5
 h5 h5) h6 (* j2 j2)) (* 84 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) h6 j2) (* 24 
(* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) h6) (* 36 (* h2 h2 h2) h3 (* h4 h4) (* h5
 h5) (* h6 h6) (* j2 j2 j2)) (* 96 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6)
 (* j2 j2)) (* 84 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 24 (* h2 
h2 h2) h3 (* h4 h4) (* h5 h5) (* h6 h6)) (* 12 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5
) h6 (* j2 j2 j2)) (* 32 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 28 
(* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 j2) (* 8 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5
) h6) (* 36 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 96 (* h2 
h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 84 (* h2 h2 h2) h3 h4 (* h5 h5
 h5) (* h6 h6) j2) (* 24 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6)) (* 24 (* h2 
h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 64 (* h2 h2 h2) h3 h4 (* h5
 h5) (* h6 h6 h6) (* j2 j2)) (* 56 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) j2)
 (* 16 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6)) (* 6 (* h2 h2 h2) h3 (* h5 h5 
h5 h5) (* h6 h6) (* j2 j2 j2)) (* 16 (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2)) (* 14 (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) j2) (* 4 (* h2 h2 h2)
 h3 (* h5 h5 h5 h5) (* h6 h6)) (* 12 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) 
(* j2 j2 j2)) (* 32 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 28 
(* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) j2) (* 8 (* h2 h2 h2) h3 (* h5 h5 h5) 
(* h6 h6 h6)) (* 6 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 16
 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 14 (* h2 h2 h2) h3 (* 
h5 h5) (* h6 h6 h6 h6) j2) (* 4 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6)) (* 36
 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 336 (* h2 h2) (* 
h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2)) (* 976 (* h2 h2) (* h3 h3 h3 h3) (* 
h4 h4 h4) h5 (* j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4 h4) h5 j2) (* 
256 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4 h4) h5) (* 72 (* h2 h2) (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (* 672 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2)) (* 1952 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) 
(* j2 j2)) (* 1792 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) j2) (* 512 (* 
h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)) (* 108 (* h2 h2) (* h3 h3 h3 h3) (* 
h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 1008 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 
(* j2 j2 j2)) (* 2928 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2)) (* 
2688 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 j2) (* 768 (* h2 h2) (* h3 h3 h3 
h3) (* h4 h4) h5 h6) (* 36 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2
 j2)) (* 336 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2)) (* 976 (* 
h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 
h3) h4 (* h5 h5 h5) j2) (* 256 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5 h5)) (* 144
 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 1344 (* h2 h2) 
(* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2)) (* 3904 (* h2 h2) (* h3 h3 h3 h3) 
h4 (* h5 h5) h6 (* j2 j2)) (* 3584 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 j2)
 (* 1024 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6) (* 108 (* h2 h2) (* h3 h3 h3
 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 1008 (* h2 h2) (* h3 h3 h3 h3) h4 h5 
(* h6 h6) (* j2 j2 j2)) (* 2928 (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 
j2)) (* 2688 (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* h6 h6) j2) (* 768 (* h2 h2) (* 
h3 h3 h3 h3) h4 h5 (* h6 h6)) (* 36 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2)) (* 336 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2)) 
(* 976 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2)) (* 896 (* h2 h2) (* 
h3 h3 h3 h3) (* h5 h5 h5) h6 j2) (* 256 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) 
h6) (* 72 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 672 
(* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 1952 (* h2 h2) 
(* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2)) (* 1792 (* h2 h2) (* h3 h3 h3 h3)
 (* h5 h5) (* h6 h6) j2) (* 512 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6)) 
(* 36 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (* 336 (* h2 h2
) (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2)) (* 976 (* h2 h2) (* h3 h3 h3 h3)
 h5 (* h6 h6 h6) (* j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6 h6) j2)
 (* 256 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6 h6)) (* 36 (* h2 h2) (* h3 h3 h3) 
(* h4 h4 h4 h4) h5 (* j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) 
h5 (* j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5 j2) (* 64 (* h2 
h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5) (* 18 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) 
(* j2 j2 j2)) (* 732 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2)) 
(* 784 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) j2) (* 256 (* h2 h2) (* h3 
h3 h3) (* h4 h4 h4) (* h5 h5)) (* 144 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 
(* j2 j2 j2)) (* 768 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2)) (* 832
 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 j2) (* 256 (* h2 h2) (* h3 h3 h3) (* 
h4 h4 h4) h5 h6) (* 18 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2
 j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 732 
(* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2)) (* 784 (* h2 h2) (* h3 
h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 256 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 
h5 h5)) (* 54 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 
630 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 2196 (* h2 h2
) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 2352 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 j2) (* 768 (* h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 h5) h6)
 (* 216 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2)) (* 1152 (* 
h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2)) (* 1248 (* h2 h2) (* h3 h3
 h3) (* h4 h4) h5 (* h6 h6) j2) (* 384 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* h6
 h6)) (* 36 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2)) (* 192 (* h2
 h2) (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) h4
 (* h5 h5 h5 h5) j2) (* 64 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5 h5)) (* 36 (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 420 (* h2 h2) (* h3 
h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 1464 (* h2 h2) (* h3 h3 h3) h4 (* h5 
h5 h5) h6 (* j2 j2)) (* 1568 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 j2) (* 
512 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6) (* 54 (* h2 h2) (* h3 h3 h3) h4 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 630 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2)) (* 2196 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) 
(* j2 j2)) (* 2352 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 768 (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6)) (* 144 (* h2 h2) (* h3 h3 h3) h4 h5 
(* h6 h6 h6) (* j2 j2 j2)) (* 768 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* 
j2 j2)) (* 832 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6 h6) j2) (* 256 (* h2 h2) 
(* h3 h3 h3) h4 h5 (* h6 h6 h6)) (* 36 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) h6
 (* j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 
208 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) h6 j2) (* 64 (* h2 h2) (* h3 h3 h3) 
(* h5 h5 h5 h5) h6) (* 18 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2
 j2 j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
732 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 784 (* h2 h2) 
(* h3 h3 h3) (* h5 h5 h5) (* h6 h6) j2) (* 256 (* h2 h2) (* h3 h3 h3) (* h5 h5 
h5) (* h6 h6)) (* 18 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 732 
(* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 784 (* h2 h2) (* h3 
h3 h3) (* h5 h5) (* h6 h6 h6) j2) (* 256 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 
h6 h6)) (* 36 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2)) (* 192 (* 
h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) 
h5 (* h6 h6 h6 h6) j2) (* 64 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6)) (* 12 
(* h2 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 68 (* h2 h2) (* 
h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2)) (* 88 (* h2 h2) (* h3 h3) (* h4 h4 
h4 h4) (* h5 h5) j2) (* 32 (* h2 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5)) (* 24 
(* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 136 (* h2 h2) (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 176 (* h2 h2) (* h3 h3) (* h4 h4 
h4) (* h5 h5 h5) j2) (* 64 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5)) (* 48 
(* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 272 (* h2 h2) (* 
h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 352 (* h2 h2) (* h3 h3) (* h4 h4 
h4) (* h5 h5) h6 j2) (* 128 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6) (* 12
 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 68 (* h2 h2) (* 
h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 88 (* h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5 h5 h5) j2) (* 32 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5)) (* 72 
(* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 408 (* h2 h2) (* 
h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 528 (* h2 h2) (* h3 h3) (* h4 h4)
 (* h5 h5 h5) h6 j2) (* 192 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6) (* 72
 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2)) (* 408 (* h2 h2
) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2)) (* 528 (* h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5) (* h6 h6) j2) (* 192 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5)
 (* h6 h6)) (* 24 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2)) (* 136
 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2)) (* 176 (* h2 h2) (* h3 h3)
 h4 (* h5 h5 h5 h5) h6 j2) (* 64 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6) (* 
72 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 408 (* h2 h2) 
(* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 528 (* h2 h2) (* h3 h3) h4 (* 
h5 h5 h5) (* h6 h6) j2) (* 192 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6)) 
(* 48 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 272 (* h2 
h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 352 (* h2 h2) (* h3 h3) h4
 (* h5 h5) (* h6 h6 h6) j2) (* 128 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6)
) (* 12 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 68 (* h2 
h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 88 (* h2 h2) (* h3 h3) (* 
h5 h5 h5 h5) (* h6 h6) j2) (* 32 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6)) 
(* 24 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 136 (* h2 
h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 176 (* h2 h2) (* h3 h3) 
(* h5 h5 h5) (* h6 h6 h6) j2) (* 64 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6
)) (* 12 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2)) (* 68 (* h2
 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 88 (* h2 h2) (* h3 h3) 
(* h5 h5) (* h6 h6 h6 h6) j2) (* 32 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6
)) (* 6 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 10 (* h2 h2) h3 
(* h4 h4 h4 h4) (* h5 h5 h5) j2) (* 4 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5 h5))
 (* 6 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 10 (* h2 h2) h3 
(* h4 h4 h4) (* h5 h5 h5 h5) j2) (* 4 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5 h5))
 (* 24 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 40 (* h2 h2) h3 
(* h4 h4 h4) (* h5 h5 h5) h6 j2) (* 16 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) h6
) (* 18 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2)) (* 30 (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5 h5) h6 j2) (* 12 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) h6
) (* 36 (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 60 (* h2 h2)
 h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* 24 (* h2 h2) h3 (* h4 h4) (* h5 h5 
h5) (* h6 h6)) (* 18 (* h2 h2) h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2)) (* 30 
(* h2 h2) h3 h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* 12 (* h2 h2) h3 h4 (* h5 h5 h5 
h5) (* h6 h6)) (* 24 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 40 
(* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 16 (* h2 h2) h3 h4 (* h5 h5 h5)
 (* h6 h6 h6)) (* 6 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2)) (* 10 
(* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* 4 (* h2 h2) h3 (* h5 h5 h5 h5) 
(* h6 h6 h6)) (* 6 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 10 
(* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6 h6) j2) (* 4 (* h2 h2) h3 (* h5 h5 h5) (* 
h6 h6 h6 h6)) (* 6 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2)) 
(* 76 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2)) (* 336 h2 (* h3 h3
 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h4 h4 h4)
 (* h5 h5) j2) (* 256 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5)) (* 6 h2 (* h3 
h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 76 h2 (* h3 h3 h3 h3) (* h4
 h4) (* h5 h5 h5) (* j2 j2 j2)) (* 336 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5)
 (* j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) j2) (* 256 h2 (* h3
 h3 h3 h3) (* h4 h4) (* h5 h5 h5)) (* 18 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) 
h6 (* j2 j2 j2 j2)) (* 228 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2
)) (* 1008 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2)) (* 1728 h2 (* h3
 h3 h3 h3) (* h4 h4) (* h5 h5) h6 j2) (* 768 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 
h5) h6) (* 12 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 152 h2 
(* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2)) (* 672 h2 (* h3 h3 h3 h3) h4 
(* h5 h5 h5) h6 (* j2 j2)) (* 1152 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 j2) (* 
512 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6) (* 18 h2 (* h3 h3 h3 h3) h4 (* h5 h5)
 (* h6 h6) (* j2 j2 j2 j2)) (* 228 h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* 
j2 j2 j2)) (* 1008 h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2)) (* 1728 
h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) j2) (* 768 h2 (* h3 h3 h3 h3) h4 (* h5
 h5) (* h6 h6)) (* 6 h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2)) 
(* 76 h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 336 h2 (* h3 h3
 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h5 h5 h5)
 (* h6 h6) j2) (* 256 h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6)) (* 6 h2 (* h3 
h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2)) (* 76 h2 (* h3 h3 h3 h3) (* h5
 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 336 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6)
 (* j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) j2) (* 256 h2 (* h3
 h3 h3 h3) (* h5 h5) (* h6 h6 h6)) (* 6 h2 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5
) (* j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2)) (* 
128 h2 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) j2) (* 64 h2 (* h3 h3 h3) (* h4 h4
 h4 h4) (* h5 h5)) (* 12 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2))
 (* 104 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 256 h2 (* h3 h3 
h3) (* h4 h4 h4) (* h5 h5 h5) j2) (* 128 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 
h5)) (* 24 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2)) (* 208 h2 (* 
h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2)) (* 512 h2 (* h3 h3 h3) (* h4 h4 
h4) (* h5 h5) h6 j2) (* 256 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6) (* 6 h2 
(* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h4
 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 128 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5)
 j2) (* 64 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5)) (* 36 h2 (* h3 h3 h3) (* 
h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2)) (* 312 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 
h5) h6 (* j2 j2)) (* 768 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 j2) (* 384 h2
 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6) (* 36 h2 (* h3 h3 h3) (* h4 h4) (* h5 
h5) (* h6 h6) (* j2 j2 j2)) (* 312 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6)
 (* j2 j2)) (* 768 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) j2) (* 384 h2 
(* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6)) (* 12 h2 (* h3 h3 h3) h4 (* h5 h5 h5
 h5) h6 (* j2 j2 j2)) (* 104 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2)) 
(* 256 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 j2) (* 128 h2 (* h3 h3 h3) h4 (* h5
 h5 h5 h5) h6) (* 36 h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 
312 h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2)) (* 768 h2 (* h3 h3 h3) 
h4 (* h5 h5 h5) (* h6 h6) j2) (* 384 h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* h6 h6)) 
(* 24 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 208 h2 (* h3 h3
 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2)) (* 512 h2 (* h3 h3 h3) h4 (* h5 h5) 
(* h6 h6 h6) j2) (* 256 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6 h6 h6)) (* 6 h2 (* h3
 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h5 h5 
h5 h5) (* h6 h6) (* j2 j2)) (* 128 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) j2)
 (* 64 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6)) (* 12 h2 (* h3 h3 h3) (* h5 h5
 h5) (* h6 h6 h6) (* j2 j2 j2)) (* 104 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6)
 (* j2 j2)) (* 256 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) j2) (* 128 h2 (* h3
 h3 h3) (* h5 h5 h5) (* h6 h6 h6)) (* 6 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6
) (* j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2)) (* 
128 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) j2) (* 64 h2 (* h3 h3 h3) (* h5 h5
) (* h6 h6 h6 h6)) (* 6 h2 (* h3 h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2)) (* 
28 h2 (* h3 h3) (* h4 h4 h4 h4) (* h5 h5 h5) j2) (* 16 h2 (* h3 h3) (* h4 h4 h4 
h4) (* h5 h5 h5)) (* 6 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2)) (* 
28 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5 h5) j2) (* 16 h2 (* h3 h3) (* h4 h4 h4)
 (* h5 h5 h5 h5)) (* 24 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2)) (* 
112 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 j2) (* 64 h2 (* h3 h3) (* h4 h4 h4
) (* h5 h5 h5) h6) (* 18 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2)) 
(* 84 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 j2) (* 48 h2 (* h3 h3) (* h4 h4)
 (* h5 h5 h5 h5) h6) (* 36 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 
j2)) (* 168 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) j2) (* 96 h2 (* h3 h3)
 (* h4 h4) (* h5 h5 h5) (* h6 h6)) (* 18 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 
h6) (* j2 j2)) (* 84 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) j2) (* 48 h2 (* 
h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6)) (* 24 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6
 h6) (* j2 j2)) (* 112 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) j2) (* 64 h2 
(* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6)) (* 6 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 
h6 h6) (* j2 j2)) (* 28 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) j2) (* 16 h2 
(* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6)) (* 6 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 
h6 h6) (* j2 j2)) (* 28 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6) j2) (* 16 h2 
(* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6))) 0)))
(check-sat)
(exit)
