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
(set-info :instance |E7E10|)
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
(+ (* 72 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 204 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 
212 (* h1 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2 j2 j2)) (* 96 (* h1
 h1 h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1 h1) 
(* h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2)) (* 12 (* h1 h1 h1 h1) (* h2 h2 h2) h3
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 38 (* h1 h1 h1 h1) (* h2 h2 h2) h3
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 (* h1 h1 h1 h1) (* h2 h2 h2) h3 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 22 (* h1 h1 h1 h1) (* h2 h2 h2) h3 (* h5 
h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* j2 
j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 7 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* (* h1 h1 h1 h1) (* h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 
72 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 708
 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1424 (* 
h1 h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2)) (* 1184 (* h1 h1 
h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 448 (* h1 h1 h1 h1) (* 
h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3
 h3) h5 (* j2 j2 j2)) (- (* 12 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 94 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 386 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 
j2 j2 j2 j2 j2 j2)) (* 488 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2
 j2 j2 j2)) (* 256 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2 j2))
 (* 48 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 12 (* h1 h1
 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 98 (* 
h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 294 
(* h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 348 
(* h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 176 (* 
h1 h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1
 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (- (* 12 (* h1 h1 h1 h1) (* 
h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 46 (* h1 h1 h1 h1) 
(* h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 234 (* h1 h1 h1 h1) 
(* h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 312 (* h1 h1 h1 h1) (* h2
 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 168 (* h1 h1 h1 h1) (* h2 h2) (* 
h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1) (* h2 h2) (* h3 h3) h5 h6
 (* j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (* 7 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 54 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 89 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2)) (* 56 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2 j2 
j2)) (* 12 (* h1 h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 4
 (* h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 10
 (* h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* 
h1 h1 h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 
h1 h1) (* h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2))) (* 12 (* h1 h1 h1 h1) (* h2
 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 38 (* h1 h1 h1 h1) (* h2
 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 (* h1 h1 h1 h1) (* h2 h2
) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 22 (* h1 h1 h1 h1) (* h2 h2) h3 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5 
h5) (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 45 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 34 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2)) (* 8 (* h1 h1 h1 h1) (* h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 4 
(* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 10 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 8 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) (- 
(* 2 (* h1 h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2))) (* 2 (* 
h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 
(* h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 
(* h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 (* 
h1 h1 h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 
h1 h1) (* h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) (* 
h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1 h1 h1) 
(* h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) 
(* h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) (* 
h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) (* h2 h2) 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) (* h2 h2) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) (* h2 h2) (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) (* h2 h2) (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) (* h2 h2) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* 12 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 62 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 
(* j2 j2 j2 j2 j2 j2 j2 j2))) (* 372 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 (* j2
 j2 j2 j2 j2 j2 j2)) (* 1448 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 (* j2 j2 j2 
j2 j2 j2)) (* 1648 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2)) (* 
768 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 128 (* h1 h1 h1 h1
) h2 (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 48 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 488 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5
 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1096 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5)
 (* j2 j2 j2 j2 j2 j2)) (* 1008 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 
j2 j2 j2 j2)) (* 416 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) 
(* 64 (* h1 h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (- (* 12 (* h1 h1 
h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 110 (* h1 h1
 h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 116 (* h1 h1 
h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2))) (* 352 (* h1 h1 h1 h1) 
h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 640 (* h1 h1 h1 h1) h2 (* h3 h3 
h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 352 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2
 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) h2 (* h3 h3 h3) h5 h6 (* j2 j2 j2)) (- (* 20 
(* h1 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 2 
(* h1 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 198 
(* h1 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 352 (* h1
 h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 224 (* h1 h1 h1 
h1) h2 (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1 h1) h2 (* h3
 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 9 (* h1 h1 h1 h1) h2 (* h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 117 (* h1 h1 h1 h1) h2 (* h3 h3) h4
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 368 (* h1 h1 h1 h1) h2 (* h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 508 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5
 h5) (* j2 j2 j2 j2 j2 j2)) (* 304 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* 
j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)
) (- (* 36 (* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
) (- (* 30 (* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 274 (* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 556 
(* h1 h1 h1 h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 368 (* h1 h1 h1 
h1) h2 (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 80 (* h1 h1 h1 h1) h2 (* h3 h3)
 h4 h5 h6 (* j2 j2 j2 j2)) (* 48 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 152 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 
j2 j2 j2 j2 j2 j2)) (* 176 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 j2 j2
 j2 j2 j2)) (* 88 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) 
(* 16 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 4 (* h1 h1 
h1 h1) h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h1 h1 
h1 h1) h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 198 (* h1 h1 h1
 h1) h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 348 (* h1 h1 h1 h1) 
h2 (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 224 (* h1 h1 h1 h1) h2 (* h3
 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1 h1) h2 (* h3 h3) (* h5 
h5) h6 (* j2 j2 j2 j2)) (- (* 16 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 32 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2))) (* 76 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 204 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) (* j2
 j2 j2 j2 j2 j2)) (* 144 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 
j2 j2)) (* 32 (* h1 h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (- (* 2
 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 7 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
)) (* 13 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 52 (* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 46 
(* h1 h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 
h1 h1) h2 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1 h1 h1) h2 
h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 10 (* h1 h1 h1 h1) h2 
h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* h1 h1 h1 h1) h2 h3 
(* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 h1 h1) h2 h3 (* h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 5 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 16 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 45 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) (* j2 
j2 j2 j2 j2 j2 j2)) (* 34 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2)) (* 8 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 4 (* 
h1 h1 h1 h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 20 (* 
h1 h1 h1 h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 71 (* h1 
h1 h1 h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 119 (* h1 h1 h1 
h1) h2 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h1 h1 h1 h1) h2 h3 
h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 20 (* h1 h1 h1 h1) h2 h3 h4 (* h5 h5) 
h6 (* j2 j2 j2 j2 j2)) (- (* 8 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2))) (- (* 20 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2))) (- (* 16 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2 j2 j2
 j2 j2 j2))) (- (* 4 (* h1 h1 h1 h1) h2 h3 h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)
)) (* 16 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 56 (* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 72 
(* h1 h1 h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 40 (* h1 h1 
h1 h1) h2 h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1 h1) h2 h3 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 49 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 34 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2)) (- (* 4 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2))) (- (* 10 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2))) (- (* 8 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) 
(- (* 2 (* h1 h1 h1 h1) h2 h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2))) (* 2 (* h1
 h1 h1 h1) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 
h1 h1 h1) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 
h1 h1) h2 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) h2
 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) h2 h4 (* h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) h2 h4 (* h5 h5 h5
) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) h2 h4 (* h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) h2 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2)) (* 2 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 14 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 9 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2))
 (* 2 (* h1 h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2 (* 
h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 7 
(* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 
(* h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* 
h1 h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 
h1) h2 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1 h1) h2 (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1 h1 h1) h2 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1 h1) h2 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1 h1 h1) h2 (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1 h1) h2 (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* 8 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 
h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 76 (* h1 h1 h1 h1) (* h3 h3 h3) 
(* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 100 (* h1 h1 h1 h1) (* h3 h3 h3
) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2))) (* 224 (* h1 h1 h1 h1) (* h3 h3 h3) 
(* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 512 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4
) h5 (* j2 j2 j2 j2 j2)) (* 320 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* j2 
j2 j2 j2)) (* 64 (* h1 h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (- (* 8
 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 76 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 100 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2))) 
(* 224 (* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 512 
(* h1 h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 320 (* h1 h1 h1
 h1) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) (* h3 h3 
h3) h4 (* h5 h5) (* j2 j2 j2)) (- (* 8 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 44 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 236 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2
 j2 j2 j2 j2 j2)) (* 1056 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 
j2 j2)) (* 1360 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 704
 (* h1 h1 h1 h1) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 128 (* h1 h1 h1 h1) 
(* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 32 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5)
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 832 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 848 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2
 j2 j2 j2)) (* 384 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) 
(* 64 (* h1 h1 h1 h1) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 32 (* h1 h1 h1 
h1) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h1 h1 h1 h1
) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 832 (* h1 h1 h1 h1) (* 
h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 848 (* h1 h1 h1 h1) (* h3 h3 h3
) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 384 (* h1 h1 h1 h1) (* h3 h3 h3) h5 (* h6 
h6) (* j2 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2
)) (- (* 8 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (- (* 20 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2
 j2))) (* 16 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2))
 (* 76 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 64 
(* h1 h1 h1 h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1 
h1) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (- (* 4 (* h1 h1 h1 h1) (* h3 h3)
 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 22 (* h1 h1 h1 h1
) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 26 (* h1 
h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 56 (* 
h1 h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 160 (* 
h1 h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 128 (* h1 
h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1
) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (- (* 16 (* h1 h1 h1 h1) (* h3 
h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* h1 h1 h1 h1) (* 
h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 144 (* h1 h1 h1 h1) (* 
h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 296 (* h1 h1 h1 h1) (* h3 h3
) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 208 (* h1 h1 h1 h1) (* h3 h3) (* h4 
h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1 h1) (* h3 h3) (* h4 h4) h5 h6 
(* j2 j2 j2 j2)) (- (* 8 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (- (* 20 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2
 j2 j2 j2 j2 j2 j2))) (* 16 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 
j2 j2 j2 j2 j2)) (* 76 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2)) (* 64 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 
16 (* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (- (* 2 (* h1 h1 
h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 3 (* 
h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 71 
(* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 300 
(* h1 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 452 (* h1
 h1 h1 h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 288 (* h1 h1 h1 
h1) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1 h1) (* h3 h3
) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 8 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 44 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 364 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 224 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2
 j2 j2 j2)) (* 48 (* h1 h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 
32 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 112
 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 144 (* 
h1 h1 h1 h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 80 (* h1 h1 h1 
h1) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1 h1) (* h3 h3
) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 8 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 (* h1 h1 h1 h1) (* h3 h3) (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 292 (* h1 h1 h1 h1) (* h3 h3) (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 160 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1) (* h3 h3) (* h5 h5) (* h6 h6
) (* j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 112 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2
 j2 j2 j2)) (* 144 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 80 (* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 16 
(* h1 h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1 
h1) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 3 (* h1 h1 
h1 h1) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 7 (* h1 h1 h1 
h1) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1 h1) h3
 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) h3 (* h4 h4 
h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 3 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 7 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 
j2 j2 j2)) (- (* 2 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (- (* 7 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (* 9 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 46 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 44 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)
) (* 12 (* h1 h1 h1 h1) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 2 
(* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
5 (* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 12 
(* h1 h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 39 (* h1 
h1 h1 h1) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1 h1) 
h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1 h1) h3 h4 (* h5 h5
 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 3 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 40 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 75 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2)) (* 52 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6) (* j2
 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2
 j2)) (* 8 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 28 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 36 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
20 (* h1 h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 
h1 h1 h1) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1 h1) h3 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 (* h1 h1 h1 h1) h3 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 36 (* h1 h1 h1 h1) h3 (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 20 (* h1 h1 h1 h1) h3 (* h5 h5)
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1 h1) h3 (* h5 h5) (* h6 h6 
h6) (* j2 j2 j2 j2 j2)) (* 36 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 j2
 j2 j2 j2 j2 j2 j2)) (* 120 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 j2 
j2 j2 j2 j2 j2)) (* 148 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2
 j2 j2)) (* 80 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 
16 (* h1 h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 (* j2 j2 j2 j2)) (* 6 (* h1 h1 h1) 
(* h2 h2 h2 h2) h3 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 22 (* h1 h1 h1) 
(* h2 h2 h2 h2) h3 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h1 h1 h1) (* 
h2 h2 h2 h2) h3 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h1 h1 h1) (* h2 h2 
h2 h2) h3 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2 h2) h3 
(* h5 h5) (* j2 j2 j2 j2 j2)) (* (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1) (* h2 h2 h2 h2) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2)) (* 72 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 816 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 1928 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2)) (* 1856
 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 800 (* h1 h1 
h1) (* h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 128 (* h1 h1 h1) (* h2 h2 
h2) (* h3 h3 h3) h5 (* j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4
 h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 296 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) 
h4 h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 914 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4
 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1028 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 
(* j2 j2 j2 j2 j2 j2)) (* 512 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2
 j2 j2 j2)) (* 96 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 
12 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 200 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2
)) (* 572 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)
) (* 656 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 336 (* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 64 
(* h1 h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (- (* 6 (* h1 h1 
h1) (* h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 200 (* h1 
h1 h1) (* h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 622 (* h1 h1
 h1) (* h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 696 (* h1 h1 h1) 
(* h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 344 (* h1 h1 h1) (* h2 h2
 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) (* h2 h2 h2) (* h3 
h3) h5 h6 (* j2 j2 j2 j2)) (- (* (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 45 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 159 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 203 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 114 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2
 j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2
 j2)) (- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))
) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2))) (- 
(* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h4 h5 h6 (* j2 j2 j2 j2 j2 j2))) (* 18 (* h1 
h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 (* h1 
h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 74 (* h1 h1 
h1) (* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 40 (* h1 h1 h1) 
(* h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1) (* h2 h2 
h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1) (* h2 h2 h2) h3 (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 100 (* h1 h1 h1) (* h2 h2 h2) h3 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 (* h1 h1 h1) (* h2 h2 h2) h3 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 74 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 
h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) (* h2 h2 h2) h3 (* h5 h5) h6 
(* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 h1) (* h2 h2 h2) h3 h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2))) (* 6 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 22 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 30 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 18 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 3 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 11 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 15 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 9 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2 
(* h1 h1 h1) (* h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1 
h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 11 
(* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 15 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 9 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 2 (* h1 h1 h1) (* h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 36 
(* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 480 
(* h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1888 (* 
h1 h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2)) (* 2496 (* h1 h1 
h1) (* h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 1344 (* h1 h1 h1) (* h2 
h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h2 h2) (* h3 h3 
h3 h3) h5 (* j2 j2 j2)) (- (* 12 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2))) (* 208 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 2628 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* 
j2 j2 j2 j2 j2 j2 j2)) (* 5880 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 
j2 j2 j2 j2 j2)) (* 5504 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 
j2 j2)) (* 2368 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 
384 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 6 (* h1 h1 h1) 
(* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 202 (* h1 h1
 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1608 (* h1
 h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 3384 (* h1
 h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 3040 (* h1 h1
 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1248 (* h1 h1 h1) 
(* h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 192 (* h1 h1 h1) (* h2 h2) 
(* h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (- (* 12 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 88 (* h1 h1 h1) (* h2 h2) (* h3 h3 
h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1360 (* h1 h1 h1) (* h2 h2) (* h3 h3 
h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3000 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) 
h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 2768 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 h6 
(* j2 j2 j2 j2 j2)) (* 1184 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 
j2 j2)) (* 192 (* h1 h1 h1) (* h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2)) (- (* 40 
(* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 234 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 1042 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)
) (* 1360 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) 
(* 736 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 144 
(* h1 h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (- (* 2 (* h1 h1 
h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 52 
(* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 608 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 1686 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)
) (* 1924 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 992 (* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 192 
(* h1 h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (- (* 80 (* h1 h1
 h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 284 (* h1 
h1 h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1544 (* h1 
h1 h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2140 (* h1 h1 
h1) (* h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 1200 (* h1 h1 h1) (* 
h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 240 (* h1 h1 h1) (* h2 h2) (* 
h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 24 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 184 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 508 (* h1 h1 h1) (* h2 h2) (* h3 h3) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 564 (* h1 h1 h1) (* h2 h2) (* h3 h3) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 272 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* 
h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5
) (* j2 j2 j2 j2)) (* (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* 287 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1004 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1300 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2)) (* 720 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2 j2)) (* 144 (* h1 h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 
j2 j2 j2)) (- (* 40 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2
 j2 j2 j2 j2 j2))) (* 50 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 502 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 780 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2)) (* 464 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 
j2 j2 j2)) (* 96 (* h1 h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) 
(- (* 6 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2 j2))) (* 14 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 138 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 240 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2)) (* 158 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2)) (* 36 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2
 j2)) (- (* 8 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 20 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2
 j2 j2))) (- (* 16 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2))) (- (* 4 (* h1 h1 h1) (* h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)
)) (- (* 3 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (* 40 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2)) (* 151 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 196 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2
)) (* 112 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24
 (* h1 h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1)
 (* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 62 (* h1 h1 
h1) (* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 271 (* h1 h1
 h1) (* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 432 (* h1 h1 
h1) (* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 274 (* h1 h1 h1) 
(* h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 60 (* h1 h1 h1) (* h2 h2)
 h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 16 (* h1 h1 h1) (* h2 h2) h3 h4 h5
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 40 (* h1 h1 h1) (* h2 h2) h3 
h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 32 (* h1 h1 h1) (* h2 h2) h3
 h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* h1 h1 h1) (* h2 h2) h3 h4
 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2))) (* 6 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) (* j2
 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 44 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 150 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2
 j2 j2 j2 j2 j2)) (* 202 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 116 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2
 j2)) (* 24 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 6 
(* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 22 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 96 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 164 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (* 108 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2))
 (* 24 (* h1 h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 8
 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 20 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 16 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) 
(- (* 4 (* h1 h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2))) (* 3 
(* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 14 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 22 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 14 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2)) (* 3 (* h1 h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 6 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 20 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 25 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 14 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3 
(* h1 h1 h1) (* h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1 
h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 
(* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 44 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 28 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 6 (* h1 h1 h1) (* h2 h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* (* 
h1 h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 
(* h1 h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 
(* h1 h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* (* h1 
h1 h1) (* h2 h2) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) 
(* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h1 
h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 23 
(* h1 h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14
 (* h1 h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3 
(* h1 h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 3 (* h1 
h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 14
 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 22 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 14 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 3 (* h1 h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* 
6 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
52 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 88 (* 
h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1424 (* h1 h1 h1
) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2)) (* 3008 (* h1 h1 h1) h2 (* h3 
h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2)) (* 2176 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 
h5 (* j2 j2 j2 j2)) (* 512 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) 
(* 24 (* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
328 (* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1360 
(* h1 h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 2016 (* h1 
h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1216 (* h1 h1 h1) h2 
(* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) h2 (* h3 h3 h3 h3
) (* h5 h5) (* j2 j2 j2)) (- (* 6 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2
 j2 j2 j2 j2 j2 j2 j2))) (- (* 76 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2
 j2 j2 j2 j2 j2 j2))) (- (* 240 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2 
j2 j2 j2 j2 j2))) (* 64 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 
j2)) (* 992 (* h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 960 (* 
h1 h1 h1) h2 (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) h2 (* h3
 h3 h3 h3) h5 h6 (* j2 j2 j2)) (- (* 44 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) 
h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 178 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4
 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 788 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 
h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 3104 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) 
h5 (* j2 j2 j2 j2 j2 j2)) (* 3440 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 (* 
j2 j2 j2 j2 j2)) (* 1568 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 
j2)) (* 256 (* h1 h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (- (* (* h1 
h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 3 (* 
h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 286 
(* h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2108 
(* h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 4336 (* 
h1 h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 3904 (* h1 h1 
h1) h2 (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1632 (* h1 h1 h1) h2 (* 
h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) h2 (* h3 h3 h3) h4 
(* h5 h5) (* j2 j2 j2)) (- (* 82 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (- (* 358 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2
 j2 j2 j2 j2 j2 j2))) (* 1424 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 6000 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2
)) (* 6816 (* h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 3136 (* 
h1 h1 h1) h2 (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 512 (* h1 h1 h1) h2 (* h3
 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 96 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 592 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2)) (* 608 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 
j2 j2)) (* 128 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (- (* 
16 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 20 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
1232 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3568
 (* h1 h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3712 (* h1 
h1 h1) h2 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1632 (* h1 h1 h1) h2 
(* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) h2 (* h3 h3 h3) 
(* h5 h5) h6 (* j2 j2 j2)) (- (* 38 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 180 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 636 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2896 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 3376 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2 j2)) (* 1568 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2))
 (* 256 (* h1 h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (- (* 34 (* h1 
h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 (* 
h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 298 (* 
h1 h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 524 (* h1 h1
 h1) h2 (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 320 (* h1 h1 h1) h2 
(* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) h2 (* h3 h3) 
(* h4 h4 h4) h5 (* j2 j2 j2 j2)) (- (* 12 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 28 (* h1 h1 h1) h2 (* h3 h3)
 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 280 (* h1 h1 h1) h2 (* 
h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1014 (* h1 h1 h1) h2 
(* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1268 (* h1 h1 h1) h2 
(* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 672 (* h1 h1 h1) h2 (* 
h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 128 (* h1 h1 h1) h2 (* h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (- (* 94 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 24 (* h1 h1 h1) h2 (* h3 h3) (* h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 970 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1636 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5
 h6 (* j2 j2 j2 j2 j2 j2)) (* 976 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6 (* 
j2 j2 j2 j2 j2)) (* 192 (* h1 h1 h1) h2 (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2
)) (- (* 4 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (* 8 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 244 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 644 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2))
 (* 712 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 352 
(* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1)
 h2 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 48 (* h1 h1 h1) h2 (* h3 h3) 
h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 568 (* h1 h1 h1) h2 (* h3 h3)
 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1982 (* h1 h1 h1) h2 (* h3 h3) 
h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2548 (* h1 h1 h1) h2 (* h3 h3) h4 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 1360 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 
h5) h6 (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1 h1) h2 (* h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2)) (- (* 86 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (* 60 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 1046 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 1700 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 992 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 192
 (* h1 h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 24 (* h1 h1 h1) 
h2 (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) h2 
(* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 56 (* h1 h1 h1) h2 (* h3 
h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) h2 (* h3 h3) (* h5
 h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 26 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 222 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6
 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 636 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2)) (* 744 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2 j2)) (* 368 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2)) (* 64 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 3 
(* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 27 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 248 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 956 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 1280 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 688 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) 
(* 128 (* h1 h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (- (* 26 
(* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 32
 (* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 374 
(* h1 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 588 (* h1
 h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 336 (* h1 h1 h1) 
h2 (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) h2 (* h3 h3)
 h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 3 (* h1 h1 h1) h2 h3 (* h4 h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 11 (* h1 h1 h1) h2 h3 (* h4 h4 h4)
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 19 (* h1 h1 h1) h2 h3 (* h4 h4 h4
) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 77 (* h1 h1 h1) h2 h3 (* h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 66 (* h1 h1 h1) h2 h3 (* h4 h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) (* j2
 j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1) h2 h3 (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2))) (- (* 4 (* h1 h1 h1) h2 h3 (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2 j2))) (- (* 2 (* h1 h1 h1) h2 h3 (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 
j2))) (- (* 6 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (- (* 14 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2
 j2 j2 j2 j2))) (* 49 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 107 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2
 j2 j2)) (* 70 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 16 (* h1 h1 h1) h2 h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 9 (* h1 
h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 138 (* h1
 h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 295 (* h1 
h1 h1) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 214 (* h1 h1 h1
) h2 h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) h2 h3 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) h2 h3 (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1 h1) h2 h3 (* h4 
h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1 h1) h2 h3 (* h4 
h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1 h1) h2 h3 h4 (* h5 h5
 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 h1) h2 h3 h4 (* h5 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 9 (* h1 h1 h1) h2 h3 h4 (* h5 h5 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2
 j2 j2 j2)) (* 2 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 49 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 195 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 270 (* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 156 
(* h1 h1 h1) h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1) 
h2 h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 9 (* h1 h1 h1) h2 h3 h4 (* h5 h5
) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 53 (* h1 h1 h1) h2 h3 h4 (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 221 (* h1 h1 h1) h2 h3 h4 (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 359 (* h1 h1 h1) h2 h3 h4 (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 230 (* h1 h1 h1) h2 h3 h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) h2 h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1 h1) h2 h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1 h1) h2 h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2))) (* 8 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 24 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 24 (* h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 8 (* 
h1 h1 h1) h2 h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1 h1) h2 
h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 49 (* h1 h1 h1) 
h2 h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 136 (* h1 h1 h1)
 h2 h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 163 (* h1 h1 h1) 
h2 h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 86 (* h1 h1 h1) h2 h3 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) h2 h3 (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 6 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 33 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 102 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 141 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 82 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) h2 h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2)) (- (* 2 (* h1 h1 h1) h2 h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2))) (- (* 4 (* h1 h1 h1) h2 h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2))) (- (* 2 (* h1 h1 h1) h2 h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) 
(* (* h1 h1 h1) h2 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
2 (* h1 h1 h1) h2 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* (* 
h1 h1 h1) h2 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 
h1) h2 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1 
h1) h2 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1) 
h2 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1) h2 (* h4 
h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1 h1) h2 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1 h1) 
h2 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1 h1) 
h2 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1) h2 h4
 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1) h2 h4 (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1) h2 h4 (* h5 h5 h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 13 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 13 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1 h1) h2 h4 (* h5 h5) (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* (* h1 h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 3 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 8 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 7 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 2 (* h1 h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))
 (* (* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 3 (* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 3 (* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
(* h1 h1 h1) h2 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (- (* 4 (* 
h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 52
 (* h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
176 (* h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2))) (* 704 
(* h1 h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 832 (* h1 h1 h1
) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3 
h3) (* h4 h4) h5 (* j2 j2 j2)) (- (* 4 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 52 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 176 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2))) (* 704 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 
h5) (* j2 j2 j2 j2 j2)) (* 832 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 
j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (- (* 
4 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
36 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 48 (* 
h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h1 h1 h1)
 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 2304 (* h1 h1 h1) (* h3 h3 
h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 1920 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 
h6 (* j2 j2 j2 j2)) (* 512 (* h1 h1 h1) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) 
(* 16 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
224 (* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 976 
(* h1 h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 1600 (* h1 
h1 h1) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1088 (* h1 h1 h1) (* 
h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3 h3) 
(* h5 h5) h6 (* j2 j2 j2)) (* 16 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 224 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 976 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2
 j2 j2 j2)) (* 1600 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2)
) (* 1088 (* h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 256 (* 
h1 h1 h1) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (- (* 16 (* h1 h1 h1) (* h3
 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 88 (* h1 h1 h1) 
(* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 40 (* h1 h1 h1)
 (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2))) (* 320 (* h1 h1 h1) (* 
h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 416 (* h1 h1 h1) (* h3 h3 h3
) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 128 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4 
h4) h5 (* j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 76 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4)
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 240 (* h1 h1 h1) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 64 (* h1 h1 h1) (* h3 h3 h3
) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 992 (* h1 h1 h1) (* h3 h3 h3)
 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 960 (* h1 h1 h1) (* h3 h3 h3) (* 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2)) (- (* 32 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 112 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) 
h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 336 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) 
h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1408 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 
h6 (* j2 j2 j2 j2 j2 j2)) (* 1376 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6 (* 
j2 j2 j2 j2 j2)) (* 384 (* h1 h1 h1) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2
)) (- (* 16 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2))) (- (* 88 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 40 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2
 j2))) (* 320 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 416 (* h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 128 (* 
h1 h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (- (* 3 (* h1 h1 h1) (* 
h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 30 (* h1 h1 
h1) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 120 (* h1 
h1 h1) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1352 (* h1 
h1 h1) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2832 (* h1 h1 
h1) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2112 (* h1 h1 h1) (* 
h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 512 (* h1 h1 h1) (* h3 h3 h3) 
h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 16 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 40 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 792 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1856 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 1504 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 
j2 j2 j2 j2)) (* 384 (* h1 h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) 
(* 64 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
416 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 768 
(* h1 h1 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 544 (* h1 h1
 h1) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 128 (* h1 h1 h1) (* h3 
h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 12 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5
) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 228 (* h1 h1 h1) (* h3 h3 h3) (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1096 (* h1 h1 h1) (* h3 h3 h3) 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1776 (* h1 h1 h1) (* h3 h3 h3) 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1152 (* h1 h1 h1) (* h3 h3 h3) (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1 h1) (* h3 h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2)) (* 64 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 416 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 768 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2)) (* 544 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2))
 (* 128 (* h1 h1 h1) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 4 (* h1
 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 
(* h1 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 12 
(* h1 h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 
h1 h1) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) 
(* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h3 h3) 
(* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 37 (* h1 h1 h1)
 (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 23 (* 
h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 132 
(* h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 188 
(* h1 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1
 h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 12 (* h1 h1 
h1) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* h1 
h1 h1) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 84 (* h1 h1
 h1) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 144 (* h1 h1 h1) 
(* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) (* h3 h3)
 (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h3 h3) (* h4 h4)
 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 37 (* h1 h1 h1) (* h3 h3
) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 23 (* h1 h1 h1) 
(* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 132 (* h1 h1 
h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 188 (* h1 h1 
h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) 
(* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (- (* 12 (* h1 h1 h1) (* h3
 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 46 (* h1 
h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 130 
(* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
584 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 
612 (* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 192 
(* h1 h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 12 (* h1
 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 24 
(* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
180 (* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
240 (* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 96 
(* h1 h1 h1) (* h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 4 (* h1 
h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 (* 
h1 h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 12 (* h1
 h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1 h1
) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) (* h3 
h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h3 h3) h4 (* 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 9 (* h1 h1 h1) (* h3 h3) 
h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 153 (* h1 h1 h1) (* h3 h3
) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 452 (* h1 h1 h1) (* h3 h3) 
h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 424 (* h1 h1 h1) (* h3 h3) h4 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 128 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5
 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6
) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 15 (* h1 h1 h1) (* h3 h3) h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 309 (* h1 h1 h1) (* h3 h3) h4 (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 756 (* h1 h1 h1) (* h3 h3) h4 (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 660 (* h1 h1 h1) (* h3 h3) h4 (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1 h1) (* h3 h3) h4 (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 40 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 156 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 176 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 
j2 j2 j2)) (* 16 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 48 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)
) (* 48 (* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 16 
(* h1 h1 h1) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1)
 (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 156 (* h1 
h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 304 (* 
h1 h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 236 (* 
h1 h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1
 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1) (* 
h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 156 (* h1 h1 h1
) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 304 (* h1 h1 
h1) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 236 (* h1 h1 
h1) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1 h1) 
(* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) (* h3 h3
) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) (* h3 h3) 
h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) (* h3 h3) h5 (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) (* h3 h3) h5 (* h6 h6 h6 
h6) (* j2 j2 j2 j2 j2)) (- (* (* h1 h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2
 j2 j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 
j2 j2 j2 j2 j2 j2 j2))) (* 4 (* h1 h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 
j2 j2)) (- (* 4 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (- (* 2 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2))) (* 16 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2
 j2)) (* 8 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- 
(* (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)))
 (- (* 4 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 5 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 24 (* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 16 
(* h1 h1 h1) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* (* h1 h1 
h1) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1
 h1) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 4 (* h1 h1 h1
) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h3 
(* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1 h1) h3 (* h4 
h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 9 (* h1 h1 h1) h3 
(* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 28 (* h1 h1 h1) h3
 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 (* h1 h1 h1) h3 
(* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1) h3 (* h4 
h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1 h1) h3 (* h4 h4) (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1 h1) h3 (* h4
 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 25 (* h1 h1 h1) h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1 h1) h3
 (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1) h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* (* h1 h1 h1) h3 h4 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1 h1) h3 
h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 7 (* h1 h1 h1) h3 h4 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1 h1) h3 h4 (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1) h3 h4 (* h5 h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2)) (- (* 3 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 11 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 62 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2)) (- (* (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (* 6 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 31 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 40 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 16 (* h1 h1 h1) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)
) (* 4 (* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 12 (* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 12 (* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 
(* h1 h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 
h1) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1 
h1) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 (* h1 h1 h1)
 h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1 h1) h3 (* 
h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h3 (* h5 h5) (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1) h3 (* h5 h5) (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1 h1) h3 (* h5 h5) (* h6 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1 h1) h3 (* h5 h5) (* h6 h6 h6 
h6) (* j2 j2 j2 j2 j2 j2)) (* 216 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* 
j2 j2 j2 j2 j2 j2 j2)) (* 648 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2
 j2 j2 j2 j2)) (* 720 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2 
j2)) (* 352 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 64 (* 
h1 h1) (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* j2 j2 j2)) (* 108 (* h1 h1) (* h2 h2 
h2 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 360 (* h1 h1) (* h2 h2 h2
 h2) (* h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 444 (* h1 h1) (* h2 h2 h2 h2) 
(* h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3
) h4 h5 (* j2 j2 j2 j2 j2)) (* 48 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h4 h5 (* 
j2 j2 j2 j2)) (* 72 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2)) (* 296 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 
j2 j2 j2)) (* 160 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2 
j2)) (* 32 (* h1 h1) (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 72 
(* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 
(* h1 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 296 (* h1
 h1) (* h2 h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 160 (* h1 h1) (* 
h2 h2 h2 h2) (* h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 32 (* h1 h1) (* h2 h2 h2 h2)
 (* h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 18 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 90 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 54 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* j2 
j2 j2 j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* j2 j2 j2 j2 
j2)) (* 6 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 22 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 30 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
18 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 
h1) (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2
 h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 (* h1 h1) (* h2 h2
 h2 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 (* h1 h1) (* h2 h2 h2
 h2) h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 36 (* h1 h1) (* h2 h2 h2 h2) 
h3 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1) (* h2 h2 h2 h2) h3 (* h5 
h5) h6 (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2 h2 h2) h4 (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2)) (* (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2))
 (* (* h1 h1) (* h2 h2 h2 h2) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* (* h1 h1
) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 
(* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 6 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 4 (* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
(* h1 h1) (* h2 h2 h2 h2) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 216 (* 
h1 h1) (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1296 (* h1 
h1) (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2 j2)) (* 2016 (* h1 h1) (* 
h2 h2 h2) (* h3 h3 h3 h3) h5 (* j2 j2 j2 j2 j2)) (* 1216 (* h1 h1) (* h2 h2 h2) 
(* h3 h3 h3 h3) h5 (* j2 j2 j2 j2)) (* 256 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3 h3
) h5 (* j2 j2 j2)) (* 180 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 2148 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 
j2 j2 j2 j2)) (* 5216 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2 
j2 j2)) (* 5168 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2)) 
(* 2304 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2)) (* 384 (* h1 
h1) (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 108 (* h1 h1) (* h2 h2 h2) 
(* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1188 (* h1 h1) (* h2 h2 
h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 2808 (* h1 h1) (* h2 h2 
h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 2720 (* h1 h1) (* h2 h2 h2)
 (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1184 (* h1 h1) (* h2 h2 h2) (* h3
 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 192 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) 
(* h5 h5) (* j2 j2 j2)) (* 108 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 1188 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* 2808 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 
j2 j2 j2 j2)) (* 2720 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 
j2)) (* 1184 (* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 192 
(* h1 h1) (* h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2)) (- (* 12 (* h1 h1) (* h2
 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 412 (* h1 h1)
 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1300 (* h1 
h1) (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1484 (* h1 
h1) (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 752 (* h1 h1) 
(* h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 144 (* h1 h1) (* h2 
h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 36 (* h1 h1) (* h2 h2 h2) (* 
h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 600 (* h1 h1) (* h2 h2 h2
) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1716 (* h1 h1) (* h2 h2
 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1968 (* h1 h1) (* h2 h2
 h2) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 1008 (* h1 h1) (* h2 h2 h2
) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 192 (* h1 h1) (* h2 h2 h2) (* h3
 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (- (* 24 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 
h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 644 (* h1 h1) (* h2 h2 h2) (* h3 h3) 
h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2072 (* h1 h1) (* h2 h2 h2) (* h3 h3) 
h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2396 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 
h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 1232 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 h6 
(* j2 j2 j2 j2 j2)) (* 240 (* h1 h1) (* h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2
 j2)) (* 12 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 176 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 484 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 
j2 j2 j2 j2)) (* 536 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 
j2 j2 j2)) (* 264 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 
j2)) (* 48 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 12 
(* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 404 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 1212 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)
) (* 1412 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 736 (* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 144 
(* h1 h1) (* h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 12 (* h1 h1
) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 232 
(* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
772 (* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
912 (* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 480 
(* h1 h1) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 96 (* h1 h1
) (* h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* h2 
h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 60 (* h1 h1
) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 220 (* 
h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 288 
(* h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 166 
(* h1 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 36 (* h1
 h1) (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1) 
(* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 
h1) (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 
h1) (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1)
 (* h2 h2 h2) h3 (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1) (* h2 
h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 45 (* h1 h1) 
(* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 159 (* h1 h1)
 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 203 (* h1 h1) 
(* h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 114 (* h1 h1) (* h2
 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) (* h2 h2 h2) 
h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 5 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 115 (* h1 h1) (* h2 h2 h2) h3 h4 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 375 (* h1 h1) (* h2 h2 h2) h3 h4
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 485 (* h1 h1) (* h2 h2 h2) h3 h4 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 280 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5
 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 60 (* h1 h1) (* h2 h2 h2) h3 h4 (* h5 h5) h6 
(* j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2))) (- (* 4 (* h1 h1) (* h2 h2 h2) h3 h4 h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2))) (* 6 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 16 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 14 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 
j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 2 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 46 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 150 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 194 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 
112 (* h1 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 24 (* h1
 h1) (* h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2
 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 45 (* h1 h1) 
(* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 143 (* h1
 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 187 (* 
h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 110 (* 
h1 h1) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1
) (* h2 h2 h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* 
h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1) 
(* h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1) 
(* h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1) (* 
h2 h2 h2) h3 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2))) (* 5 (* h1 h1) (* h2 h2 h2)
 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h1 h1) (* h2
 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) 
(* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h1 h1)
 (* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) 
(* h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1) (* h2 
h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h1 h1) (* 
h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) (* 
h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h1 h1) (* h2 
h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) 
h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 9 (* h1 h1) (* h2 h2 h2) h4 (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 33 (* h1 h1) (* h2 h2 h2) h4
 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 45 (* h1 h1) (* h2 h2 h2
) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 27 (* h1 h1) (* h2 h2 
h2) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2 h2)
 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* (* h1 h1) (* h2 h2 h2) (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1) (* h2 h2 h2) (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2) (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 15 (* h1 h1) (* h2 h2 h2) (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 21 (* h1 h1) (* h2 h2 h2) (* h5
 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 13 (* h1 h1) (* h2 h2 h2) (* 
h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2 h2) (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 15 (* h1 h1) (* h2 h2 h2) (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 21 (* h1 h1) (* h2 h2 h2)
 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 13 (* h1 h1) (* h2 h2 h2
) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2 h2) 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) (* h2 h2) (* h3 h3
 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1248 (* h1 h1) (* h2 h2) (* h3 h3 
h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 5600 (* h1 h1) (* h2 h2) (* h3 h3 h3 
h3) h4 h5 (* j2 j2 j2 j2 j2 j2)) (* 8192 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 
h5 (* j2 j2 j2 j2 j2)) (* 4864 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 
j2 j2 j2)) (* 1024 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 36
 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
624 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
2800 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 
4096 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 2432 
(* h1 h1) (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 512 (* h1 h1) 
(* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2)) (* 36 (* h1 h1) (* h2 h2) (* 
h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 624 (* h1 h1) (* h2 h2) (* h3
 h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2800 (* h1 h1) (* h2 h2) (* h3 h3 
h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 4096 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) 
h5 h6 (* j2 j2 j2 j2 j2)) (* 2432 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 h6 (* 
j2 j2 j2 j2)) (* 512 (* h1 h1) (* h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2)) (- 
(* 24 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (* 272 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 3168 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 
j2 j2)) (* 6784 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 
j2)) (* 6128 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) 
(* 2528 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 384 
(* h1 h1) (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 24 (* h1 h1) (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 640 (* h1 h1
) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4160 (* h1
 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 8112 (* h1
 h1) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 6864 (* h1 h1
) (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 2656 (* h1 h1) (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 384 (* h1 h1) (* h2 h2) (* 
h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (- (* 48 (* h1 h1) (* h2 h2) (* h3 h3 h3) 
h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 364 (* h1 h1) (* h2 h2) (* h3 h3 h3
) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 5268 (* h1 h1) (* h2 h2) (* h3 h3 h3)
 h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 11952 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4
 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 11312 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6
 (* j2 j2 j2 j2 j2)) (* 4864 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 
j2 j2)) (* 768 (* h1 h1) (* h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 6 (* 
h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
214 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 1148 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 1824 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 
1152 (* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 256 
(* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (- (* 6 (* h1 h1)
 (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 278 (* 
h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2768 
(* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 6160 
(* h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 5792 (* 
h1 h1) (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 2464 (* h1 h1)
 (* h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 384 (* h1 h1) (* h2 h2)
 (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (- (* 24 (* h1 h1) (* h2 h2) (* h3 h3 
h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 92 (* h1 h1) (* h2 h2) (* 
h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2100 (* h1 h1) (* h2 h2) 
(* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 5168 (* h1 h1) (* h2 h2) 
(* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 5184 (* h1 h1) (* h2 h2) (* 
h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 2336 (* h1 h1) (* h2 h2) (* h3 h3 
h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 384 (* h1 h1) (* h2 h2) (* h3 h3 h3) h5 (* 
h6 h6) (* j2 j2 j2)) (- (* 34 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 190 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) 
h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 824 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4
) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1032 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4
) h5 (* j2 j2 j2 j2 j2 j2)) (* 528 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5
 (* j2 j2 j2 j2 j2)) (* 96 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 
j2 j2 j2)) (- (* 6 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2 j2))) (* 70 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 928 (* h1 h1) (* h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2328 (* h1 h1) (* h2 h2) (* h3 h3) (* 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 2400 (* h1 h1) (* h2 h2) (* h3 h3)
 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 1120 (* h1 h1) (* h2 h2) (* h3 h3
) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 192 (* h1 h1) (* h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (- (* 102 (* h1 h1) (* h2 h2) (* h3 h3) (* 
h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 514 (* h1 h1) (* h2 h2) (* h3 
h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2348 (* h1 h1) (* h2 h2) (* 
h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3012 (* h1 h1) (* h2 h2) (* 
h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 1568 (* h1 h1) (* h2 h2) (* h3 
h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 288 (* h1 h1) (* h2 h2) (* h3 h3) (* 
h4 h4) h5 h6 (* j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 62 (* h1 h1) (* h2 h2) (* h3 h3) h4 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 564 (* h1 h1) (* h2 h2) (* h3 h3
) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1360 (* h1 h1) (* h2 h2) (* h3
 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1344 (* h1 h1) (* h2 h2) (* h3
 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 592 (* h1 h1) (* h2 h2) (* h3 h3)
 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 96 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 140 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5
 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 1493 (* h1 h1) (* h2 h2) (* h3 h3) h4
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4040 (* h1 h1) (* h2 h2) (* h3 h3)
 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4468 (* h1 h1) (* h2 h2) (* h3 h3)
 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2176 (* h1 h1) (* h2 h2) (* h3 h3) h4
 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 384 (* h1 h1) (* h2 h2) (* h3 h3) h4 (* h5 
h5) h6 (* j2 j2 j2 j2)) (- (* 102 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 458 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2224 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2928 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1552 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* 
h6 h6) (* j2 j2 j2 j2 j2)) (* 288 (* h1 h1) (* h2 h2) (* h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 58 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2
 j2 j2 j2 j2 j2 j2)) (* 144 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 
j2 j2 j2 j2 j2 j2)) (* 120 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 
j2 j2 j2 j2 j2)) (* 32 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2
 j2 j2)) (* (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 56 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2 j2)) (* 439 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 1140 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* 1228 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2)) (* 576 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2)) (* 96 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* 3 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 46 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 571 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 1736 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2068 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1056 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 192 (* h1 h1) (* h2 h2) (* h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2)) (- (* 34 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6 h6
) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 134 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6
 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 700 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 948 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 512 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 
h6 h6) (* j2 j2 j2 j2 j2)) (* 96 (* h1 h1) (* h2 h2) (* h3 h3) h5 (* h6 h6 h6) 
(* j2 j2 j2 j2)) (- (* 5 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 11 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 109 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 183 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 114 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* 
h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) (* h5 h5
) (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) h5 h6 (* j2 j2
 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) h5 h6 (* j2
 j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* h1 h1) (* h2 h2) h3 (* h4 h4 h4) h5 h6 (* j2
 j2 j2 j2 j2 j2 j2))) (- (* 5 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 53 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 191 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 231 (* h1 h1) (* h2 h2) h3 (* h4 h4) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 122 (* h1 h1) (* h2 h2) h3 (* h4 h4) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5
 h5 h5) (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 372 (* h1 h1) (* h2 h2) h3 (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 580 (* h1 h1) (* h2 h2) h3 (* h4 h4
) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 350 (* h1 h1) (* h2 h2) h3 (* h4 h4)
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) (* h2 h2) h3 (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1) (* h2 h2) h3 (* h4 h4) h5 (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2) h3 (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1) (* h2 h2) h3 (* h4 
h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2))) (- (* (* h1 h1) (* h2 h2) h3 h4 (* 
h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 16 (* h1 h1) (* h2 h2) h3 h4
 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 51 (* h1 h1) (* h2 h2) h3 h4
 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 50 (* h1 h1) (* h2 h2) h3 h4 
(* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1) (* h2 h2) h3 h4 (* h5 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 127 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5)
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 397 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5
) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 481 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 252 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2)) (* 16 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2
 j2 j2 j2 j2 j2)) (* 105 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 410 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 607 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 358 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) (* h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2
 j2 j2 j2 j2)) (- (* 6 (* h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1) (* h2 h2) h3 h4 h5 (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2))) (* (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 20 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 55 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 52 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 16 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 9 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
 j2)) (* 68 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 199 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 246 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 130 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2)) (* 24 (* h1 h1) (* h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2
)) (* 9 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2 j2)) (* 44 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 147 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 210 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2)) (* 122 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2)) (* 24 (* h1 h1) (* h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
 j2)) (- (* 2 (* h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 4 (* h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 2 (* h1 h1) (* h2 h2) h3 h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2))) (* (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 5 (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 5 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 14 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2
 j2 j2 j2 j2 j2 j2)) (* 13 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2) (* h4 h4) (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 15 (* h1 h1) (* h2 h2) (* h4 h4) (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2) (* h4 h4) 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2) h4 (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2) h4 (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2 h2) h4 (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2) h4 (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 9 (* h1 h1) (* h2 h2) h4 (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 26 (* h1 h1) (* h2 h2) h4 (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 25 (* h1 h1) (* h2 h2) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1) (* h2 h2) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) (* h2 h2) h4 (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1) (* h2 h2)
 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 15 (* h1 h1) (* h2
 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) (* h2
 h2) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) (* h2 h2
) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 h1) 
(* h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h1 
h1) (* h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2 (* h1 
h1) (* h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1)
 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 
(* h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 12 (* h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 4 (* h1 h1) (* h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 4 (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 5 (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 2 (* h1 h1) (* h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (- (* 12 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 104 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 
j2 j2 j2))) (* 80 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2
 j2)) (* 2208 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) 
(* 4768 (* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 3392 
(* h1 h1) h2 (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 768 (* h1 h1) h2 
(* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 6 (* h1 h1) h2 (* h3 h3 h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 124 (* h1 h1) h2 (* h3 h3 h3 h3) h4
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1088 (* h1 h1) h2 (* h3 h3 h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 3936 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* 
h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 5536 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) 
(* j2 j2 j2 j2 j2)) (* 3392 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 
j2 j2)) (* 768 (* h1 h1) h2 (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (- (* 24 
(* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 208
 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 160 (* 
h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4416 (* h1 h1) 
h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 9536 (* h1 h1) h2 (* h3 h3
 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 6784 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 
h6 (* j2 j2 j2 j2)) (* 1536 (* h1 h1) h2 (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) 
(* 24 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
232 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 720 
(* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 768 (* h1 h1
) h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1) h2 (* h3 
h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (- (* 6 (* h1 h1) h2 (* h3 h3 h3 h3) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 28 (* h1 h1) h2 (* h3 h3 h3 h3)
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 416 (* h1 h1) h2 (* h3 h3 h3 h3) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2784 (* h1 h1) h2 (* h3 h3 h3 h3) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 5024 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2)) (* 3392 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 
j2 j2 j2)) (* 768 (* h1 h1) h2 (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (- (* 
12 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 104 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
) (* 80 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
2208 (* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4768 
(* h1 h1) h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 3392 (* h1 h1) 
h2 (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 768 (* h1 h1) h2 (* h3 h3 h3
 h3) h5 (* h6 h6) (* j2 j2 j2)) (- (* 38 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) 
h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 80 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 
h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 672 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 
h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1888 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) 
h5 (* j2 j2 j2 j2 j2 j2)) (* 1504 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) h5 (* 
j2 j2 j2 j2 j2)) (* 384 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2
)) (- (* 6 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2 j2))) (- (* 28 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (* 454 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 2916 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 4928 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2 j2)) (* 3328 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 
j2 j2 j2 j2)) (* 768 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 
j2)) (- (* 108 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2
 j2 j2))) (- (* 202 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2 j2))) (* 2096 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 5776 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2))
 (* 4576 (* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 1152 
(* h1 h1) h2 (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (- (* (* h1 h1) h2 
(* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* (* h1 h1) h2
 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 286 (* h1 h1) 
h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1600 (* h1 h1) 
h2 (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 2528 (* h1 h1) h2 
(* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 1632 (* h1 h1) h2 (* h3 
h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 384 (* h1 h1) h2 (* h3 h3 h3) h4 
(* h5 h5 h5) (* j2 j2 j2 j2)) (- (* (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 9 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 908 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5)
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 5468 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 9744 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 6656 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 
j2 j2 j2 j2)) (* 1536 (* h1 h1) h2 (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2))
 (- (* 102 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (- (* 164 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2))) (* 2176 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (* 5888 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(* 4640 (* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 1152 
(* h1 h1) h2 (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 24 (* h1 h1) h2 
(* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 136 (* h1 h1) h2 
(* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 176 (* h1 h1) h2 (* h3
 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) h2 (* h3 h3 h3) 
(* h5 h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 2 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 228 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5)
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1408 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2512 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 1696 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2)) (* 384 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) 
(* (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 3 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 370 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 2488 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 4752 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 3328 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)
) (* 768 (* h1 h1) h2 (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (- (* 32
 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 42 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 752 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
2000 (* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1568 
(* h1 h1) h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 384 (* h1 h1) 
h2 (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 10 (* h1 h1) h2 (* h3 h3)
 (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 6 (* h1 h1) h2 (* h3 h3)
 (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 (* h1 h1) h2 (* h3 h3) 
(* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 128 (* h1 h1) h2 (* h3 h3) (* h4
 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4 h4
) h5 (* j2 j2 j2 j2 j2)) (- (* 11 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 23 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4)
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 266 (* h1 h1) h2 (* h3 h3) (* h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 770 (* h1 h1) h2 (* h3 h3) (* 
h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 684 (* h1 h1) h2 (* h3 h3) (* 
h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) h2 (* h3 h3) (* h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 38 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 38 (* h1 h1) h2 (* h3 h3) (* h4 h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 412 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 528 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 
h6 (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) h2 (* h3 h3) (* h4 h4 h4) h5 h6 (* j2
 j2 j2 j2 j2)) (- (* 11 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 7 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 394 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 866 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 700 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5 
h5) (* j2 j2 j2 j2 j2)) (- (* 14 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 33 (* h1 h1) h2 (* h3 h3) (* h4 h4) (* h5
 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 905 (* h1 h1) h2 (* h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2386 (* h1 h1) h2 (* h3 h3) (* h4 
h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2068 (* h1 h1) h2 (* h3 h3) (* h4 
h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 576 (* h1 h1) h2 (* h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 54 (* h1 h1) h2 (* h3 h3) (* h4 h4) h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 78 (* h1 h1) h2 (* h3 h3) (* h4 h4
) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 660 (* h1 h1) h2 (* h3 h3) (* h4 
h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 816 (* h1 h1) h2 (* h3 h3) (* h4 
h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 288 (* h1 h1) h2 (* h3 h3) (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 3 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 78 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 180 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2)) (* 160 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 
j2 j2)) (- (* 3 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2))) (* 104 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 839 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 1776 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 1416 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2)) (* 384 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 2 
(* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 120 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2)) (* 996 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 2446 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 2084 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 576 (* h1 h1) h2 (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) 
(- (* 34 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 66 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 468 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
560 (* h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 (* 
h1 h1) h2 (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 8 (* h1 h1) h2 (* 
h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 88 (* h1 h1) h2 (* 
h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 208 (* h1 h1) h2 (* h3
 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 176 (* h1 h1) h2 (* h3 h3) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1) h2 (* h3 h3) (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 5 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 82 (* h1 h1) h2 (* h3 h3) (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 429 (* h1 h1) h2 (* h3 h3) (* h5 h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 894 (* h1 h1) h2 (* h3 h3) (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 716 (* h1 h1) h2 (* h3 h3) (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) h2 (* h3 h3) (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 5 (* h1 h1) h2 (* h3 h3) (* h5 h5) (* h6 h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) h2 (* h3 h3) (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 357 (* h1 h1) h2 (* h3 h3) (* h5
 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 830 (* h1 h1) h2 (* h3 h3) (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 700 (* h1 h1) h2 (* h3 h3) (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) h2 (* h3 h3) (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (- (* 8 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 20 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6
 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 124 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 144 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6 h6
) (* j2 j2 j2 j2 j2 j2)) (* 48 (* h1 h1) h2 (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 
j2 j2 j2 j2)) (- (* (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (- (* 3 (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (* 8 (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 22 (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 
j2 j2 j2)) (* 12 (* h1 h1) h2 h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)
) (- (* 5 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 9 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (* 46 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2
 j2 j2)) (* 68 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2
)) (* 24 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 
(* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
54 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 100
 (* h1 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 48 (* h1
 h1) h2 h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1) h2
 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 (* h1 
h1) h2 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 18 (* h1
 h1) h2 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h1 h1
) h2 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1) h2 
h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 3 (* h1 h1) h2 h3 (* 
h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 31 (* h1 h1) h2 h3
 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 196 (* h1 h1) h2 
h3 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 220 (* h1 h1) h2 h3
 (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) h2 h3 (* h4
 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) h2 h3 (* h4 h4) (* h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 21 (* h1 h1) h2 h3 (* h4 h4
) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 114 (* h1 h1) h2 h3 (* 
h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 168 (* h1 h1) h2 h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) h2 h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h1 h1) h2 h3 h4 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 (* h1 h1) h2 h3 h4 (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) h2 h3 h4 (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) h2 h3 h4 (* h5 h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 7 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 79 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 246 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 236 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 72 (* h1 h1) h2 h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2)) (* 5 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 27 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 98 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 124 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2)) (* 48 (* h1 h1) h2 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 
2 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 20 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 48 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
42 (* h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 (* 
h1 h1) h2 h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1) h2 
h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 39 (* h1 h1) 
h2 h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 (* h1 h1) 
h2 h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h1 h1) h2 
h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h1 h1) h2 h3 (* 
h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) h2 h3 (* h5 h5) (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 10 (* h1 h1) h2 h3 (* h5 h5) 
(* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h1 h1) h2 h3 (* h5 h5) 
(* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 34 (* h1 h1) h2 h3 (* h5 h5) (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h1 h1) h2 h3 (* h5 h5) (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2
 j2 j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2 j2)) (* (* h1 h1) h2 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 4 (* h1 h1) h2 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) h2 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* (* h1 h1) h2 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) h2 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 2 (* h1 h1) h2 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 2 (* h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 5 (* h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 (* h1 h1) h2 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 2 (* h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* (* h1 h1) h2 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* (* h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 2 (* h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* (* h1 h1) h2 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (- (* 4 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (- (* 36 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2
 j2 j2))) (- (* 80 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 
j2 j2))) (* 80 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) 
(* 384 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 256 (* 
h1 h1) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* h3 
h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 32 (* h1
 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
160 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 192 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)
)) (* 544 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 1280 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 512
 (* h1 h1) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (- (* 8 (* h1 h1
) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 56 (* 
h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 688 (* 
h1 h1) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 1408 (* h1 h1) 
(* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 768 (* h1 h1) (* h3 h3 h3
 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5
 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 36 (* h1 h1) (* h3 h3 h3 h3) h4 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 80 (* h1 h1) (* h3 h3 h3 h3) h4
 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2))) (* 80 (* h1 h1) (* h3 h3 h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 384 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5 h5
) (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 
j2 j2)) (- (* (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (- (* 12 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2 j2 j2))) (* 20 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 624 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2)) (* 2304 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)
) (* 2816 (* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1024 
(* h1 h1) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* 
h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 (* h1 h1) 
(* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 240 (* h1 h1) 
(* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1136 (* h1 h1) (* 
h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1664 (* h1 h1) (* h3 h3 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 768 (* h1 h1) (* h3 h3 h3 h3) h4 
h5 (* h6 h6) (* j2 j2 j2 j2)) (* 16 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 160 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2)) (* 528 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2 j2)) (* 640 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 4 
(* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 84 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* 560 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 1504 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 
1536 (* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 512 
(* h1 h1) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 16 (* h1 h1) 
(* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 160 (* h1 h1) 
(* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 528 (* h1 h1) (* h3
 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 640 (* h1 h1) (* h3 h3 h3 
h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 20 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5
 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 80 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 
j2 j2 j2)) (- (* 6 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (- (* 55 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 116 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) (* 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 156 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 560 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) (* 
h5 h5) (* j2 j2 j2 j2 j2)) (- (* 12 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 h6 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 44 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 96 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 384 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 h6 
(* j2 j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2
 j2 j2 j2)) (- (* 6 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2
 j2 j2 j2 j2 j2 j2))) (- (* 55 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 116 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 156 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 560 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2)) (- (* 11 (* h1 h1) (* h3 h3 h3) (* h4 h4) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 78 (* h1 h1) (* h3 h3 h3) (* h4 h4)
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 24 (* h1 h1) (* h3 h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1060 (* h1 h1) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1872 (* h1 h1) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 768 (* h1 h1) (* h3 h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 12 (* h1 h1) (* h3 h3 h3) (* h4 h4
) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 (* h1 h1) (* h3 h3 h3) 
(* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 288 (* h1 h1) (* h3 h3 
h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 672 (* h1 h1) (* h3 h3 
h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 384 (* h1 h1) (* h3 h3 h3) 
(* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* h3 h3 h3) h4 (* 
h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 20 (* h1 h1) (* h3 h3 h3) h4
 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 80 (* h1 h1) (* h3 h3 h3) h4 
(* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 
h5 h5) (* j2 j2 j2 j2 j2)) (- (* 5 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 23 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5)
 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 140 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 904 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5
) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1312 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6
 (* j2 j2 j2 j2 j2 j2)) (* 512 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2)) (- (* 5 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (- (* 3 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 348 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1588 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2064 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 768 (* h1 h1) (* h3 h3 h3) h4 (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2)) (- (* 4 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 28 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 288 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 512 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2
 j2 j2 j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 
j2 j2)) (* 16 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 96 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 144 (* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 64 
(* h1 h1) (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 20 (* h1 h1) 
(* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 208 (* h1
 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 684 (* 
h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 752 (* 
h1 h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h1 
h1) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 20 (* h1 h1) (* 
h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 208 (* h1 h1
) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 684 (* h1 
h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 752 (* h1 
h1) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h1 h1) 
(* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 16 (* h1 h1) (* h3 h3
 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 (* h1 h1) (* h3 h3 h3
) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 144 (* h1 h1) (* h3 h3 h3) h5 
(* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) (* h3 h3 h3) h5 (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 10 (* h1 h1) (* h3 h3) (* h4 h4 h4 h4
) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 40 (* h1 h1) (* h3 h3) (* h4 h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h1 h1) (* h3 h3) (* h4 h4 h4
 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1) (* h3 h3) (* h4 h4 h4) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 28 (* h1 h1) (* h3 h3) 
(* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 8 (* h1 h1) (* h3 
h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 112 (* h1 h1) (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h1 h1) (* h3
 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1) (* h3 h3
) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 22 (* h1 
h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 48 
(* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
192 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 
128 (* h1 h1) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 2
 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)
)) (- (* 10 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (* 40 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2)) (* 32 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2
 j2)) (- (* 11 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (- (* 28 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 148 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 384 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2)) (- (* 6 (* h1 h1) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 6 (* h1 h1) (* h3 h3) (* h4 h4) 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 144 (* h1 h1) (* h3 h3) 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h1 h1) (* 
h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 192 (* h1 h1) 
(* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* 2 (* h1 h1)
 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 2 (* 
h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 48 
(* h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 112 
(* h1 h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h1 
h1) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 5 (* h1 h1) (* 
h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 20 (* h1
 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 256 
(* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
432 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
192 (* h1 h1) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* 2
 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)
)) (* 14 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 144 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 256 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 128 (* h1 h1) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 8 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 48 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2
 j2 j2 j2)) (* 72 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 32 (* h1 h1) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2
 j2)) (* 20 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 116 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 160 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 64 (* h1 h1) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
 j2 j2)) (* 8 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 48 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 72 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2
 j2 j2 j2)) (* 32 (* h1 h1) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (- (* (* h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (* 4 (* h1 h1) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)
) (- (* (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
)) (* 4 (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (- 
(* (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)))
 (- (* 3 (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 8 (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 16 (* h1 h1) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (- (* 
(* h1 h1) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 2 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
)) (* 8 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 12 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (- (* 
2 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 24 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 24 (* h1 h1) h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (- (* (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2))) (* 3 (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 16 (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 12 (* h1 h1) h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(- (* (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 7 (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
)) (* 24 (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 16 (* h1 h1) h3 h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 
(* h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 
(* h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* 
h1 h1) h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1) 
h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 (* h1 h1) 
h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h1 h1) h3 
(* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 432 h1 (* h2 h2 h2 h2)
 (* h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 1296 h1 (* h2 h2 h2 h2) (* h3 
h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2)) (* 1440 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 
h5 (* j2 j2 j2 j2 j2)) (* 704 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2 
j2)) (* 128 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 (* j2 j2 j2)) (* 216 h1 (* h2 
h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 648 h1 (* h2 h2 h2
 h2) (* h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 720 h1 (* h2 h2 h2 h2) (* 
h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 352 h1 (* h2 h2 h2 h2) (* h3 h3 h3) 
(* h5 h5) (* j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* j2
 j2 j2)) (* 216 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 648 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 720 h1 
(* h2 h2 h2 h2) (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) (* 352 h1 (* h2 h2 h2 h2)
 (* h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2 h2) (* h3 h3 h3) h5 h6
 (* j2 j2 j2)) (* 108 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 360 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2
 j2 j2)) (* 444 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2))
 (* 240 h1 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 48 h1 
(* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 144 h1 (* h2 h2 h2 h2
) (* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 480 h1 (* h2 h2 h2 h2) 
(* h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 592 h1 (* h2 h2 h2 h2) (* h3
 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 320 h1 (* h2 h2 h2 h2) (* h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) 
(* j2 j2 j2 j2)) (* 180 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 600 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)
) (* 740 h1 (* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 400 h1 
(* h2 h2 h2 h2) (* h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 80 h1 (* h2 h2 h2 h2) 
(* h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 36 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 120 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2)) (* 148 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2)) (* 80 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 
j2 j2)) (* 16 h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 108 
h1 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 360 h1
 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 444 h1 (* 
h2 h2 h2 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 240 h1 (* h2 h2 h2
 h2) (* h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2 h2 h2) (* h3 
h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 72 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 296 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2
 j2 j2 j2 j2 j2)) (* 160 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 
j2 j2)) (* 32 h1 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 18 
h1 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 
h1 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 90 h1 
(* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 54 h1 (* h2 
h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2 h2)
 h3 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2 h2) h3 h4 (* h5
 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 
j2 j2 j2)) (* 8 h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 30 
h1 (* h2 h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 110 h1 
(* h2 h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 150 h1 (* h2 
h2 h2 h2) h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 90 h1 (* h2 h2 h2 h2) 
h3 h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 20 h1 (* h2 h2 h2 h2) h3 h4 (* h5 
h5) h6 (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 44 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 60 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 8 
h1 (* h2 h2 h2 h2) h3 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2 
h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 h1 (* h2 h2 h2 
h2) h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 h1 (* h2 h2 h2 h2)
 h3 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 36 h1 (* h2 h2 h2 h2) h3 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 8 h1 (* h2 h2 h2 h2) h3 (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2 h2) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2)) (* h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 4 h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 6 h1 (* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 
(* h2 h2 h2 h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2 
h2) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2 h2 h2) h4 (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 (* h2 h2 h2 h2) h4 (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2 h2) h4 (* 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 (* h2 h2 h2 h2) h4 (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2 h2 h2) h4 (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2 h2) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 6 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 4 h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* h1 (* h2 h2 h2 h2) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 432 h1 
(* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 2592 h1 (* h2 h2
 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2 j2)) (* 4032 h1 (* h2 h2 h2) (* h3 
h3 h3 h3) h4 h5 (* j2 j2 j2 j2 j2)) (* 2432 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 
h5 (* j2 j2 j2 j2)) (* 512 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 (* j2 j2 j2)) 
(* 216 h1 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
1296 h1 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 2016 h1
 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1216 h1 (* h2 h2 
h2) (* h3 h3 h3 h3) (* h5 h5) (* j2 j2 j2 j2)) (* 256 h1 (* h2 h2 h2) (* h3 h3 
h3 h3) (* h5 h5) (* j2 j2 j2)) (* 216 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* 
j2 j2 j2 j2 j2 j2 j2)) (* 1296 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2
 j2 j2 j2)) (* 2016 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2 j2)) 
(* 1216 h1 (* h2 h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2 j2)) (* 256 h1 (* h2 
h2 h2) (* h3 h3 h3 h3) h5 h6 (* j2 j2 j2)) (* 180 h1 (* h2 h2 h2) (* h3 h3 h3) 
(* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1788 h1 (* h2 h2 h2) (* h3 h3 h3) 
(* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 4088 h1 (* h2 h2 h2) (* h3 h3 h3) (* 
h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 3856 h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) 
h5 (* j2 j2 j2 j2 j2)) (* 1632 h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 
j2 j2 j2)) (* 256 h1 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 216
 h1 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2016 
h1 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 4488 h1 
(* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4128 h1 (* h2 
h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 1696 h1 (* h2 h2 h2) (* 
h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 256 h1 (* h2 h2 h2) (* h3 h3 h3) h4 
(* h5 h5) (* j2 j2 j2)) (* 288 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2
 j2 j2 j2 j2 j2)) (* 3120 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 7376 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) 
(* 7168 h1 (* h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 3136 h1 (* 
h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 512 h1 (* h2 h2 h2) (* h3 h3
 h3) h4 h5 h6 (* j2 j2 j2)) (* 72 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 456 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2 j2)) (* 800 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 
j2 j2 j2 j2)) (* 544 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2
)) (* 128 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2)) (* 144 h1 
(* h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1560 h1 
(* h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3688 h1 (* 
h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3584 h1 (* h2 h2 
h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1568 h1 (* h2 h2 h2) (* h3 
h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 256 h1 (* h2 h2 h2) (* h3 h3 h3) (* h5 
h5) h6 (* j2 j2 j2)) (* 108 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 1332 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2)) (* 3288 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 
j2 j2 j2)) (* 3312 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2))
 (* 1504 h1 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2)) (* 256 h1 
(* h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (- (* 6 h1 (* h2 h2 h2) (* 
h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 224 h1 (* h2 h2 h2) 
(* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 674 h1 (* h2 h2 h2) 
(* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 732 h1 (* h2 h2 h2) (* h3
 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 352 h1 (* h2 h2 h2) (* h3 h3) (* 
h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5
 (* j2 j2 j2 j2)) (* 42 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 556 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 1462 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 1556 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) 
(* j2 j2 j2 j2 j2 j2)) (* 736 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2)) (* 128 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 
j2 j2)) (- (* 18 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (* 636 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 1938 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 2132 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2))
 (* 1040 h1 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 192 h1
 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 24 h1 (* h2 h2 h2) 
(* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 316 h1 (* h2 h2 h2)
 (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 812 h1 (* h2 h2 h2) 
(* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 840 h1 (* h2 h2 h2) (* h3
 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 384 h1 (* h2 h2 h2) (* h3 h3) h4 
(* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5)
 (* j2 j2 j2 j2)) (* 66 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2 j2)) (* 992 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 2690 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 2932 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2)) (* 1424 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) 
(* 256 h1 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 18 h1 
(* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 600 
h1 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1854 
h1 (* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2068 h1 
(* h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1024 h1 (* h2 
h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2 h2) (* 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 36 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5
 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* j2 j2 
j2 j2 j2)) (* 24 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2 j2)) (* 280 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2
 j2 j2)) (* 728 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2)) (* 776 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 
368 h1 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 64 h1 (* h2
 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 24 h1 (* h2 h2 h2) (* h3 
h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 436 h1 (* h2 h2 h2) 
(* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1228 h1 (* h2 h2 
h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1376 h1 (* h2 h2 
h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 688 h1 (* h2 h2 h2) 
(* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 128 h1 (* h2 h2 h2) (* h3 
h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (- (* 6 h1 (* h2 h2 h2) (* h3 h3) h5 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 188 h1 (* h2 h2 h2) (* h3 h3) 
h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 590 h1 (* h2 h2 h2) (* h3 h3) h5
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 668 h1 (* h2 h2 h2) (* h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 336 h1 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6
) (* j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2 
j2 j2)) (- (* h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (* 33 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 115 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 143 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2
 j2)) (* 78 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 
16 h1 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* h1 (* h2 
h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 39 h1 
(* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 131 h1
 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 157 h1 
(* h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 82 h1 (* h2 
h2 h2) h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h2 h2 h2) h3
 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2) h3 (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 103 h1 (* h2 h2 h2) h3 (* h4 h4)
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 341 h1 (* h2 h2 h2) h3 (* h4 h4
) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 425 h1 (* h2 h2 h2) h3 (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 234 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5
 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 
(* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 32 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 28 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2))
 (* 8 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2
 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 82 h1 (* h2 
h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 258 h1 (* h2 h2 
h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 310 h1 (* h2 h2 h2) h3
 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 164 h1 (* h2 h2 h2) h3 h4 (* h5
 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 32 h1 (* h2 h2 h2) h3 h4 (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2)) (* 5 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 107 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 337 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 421 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2
 j2 j2 j2 j2 j2)) (* 234 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2)) (* 48 h1 (* h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) 
(* 12 h1 (* h2 h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
32 h1 (* h2 h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 h1 
(* h2 h2 h2) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 (* h2 h2 h2
) h3 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) h3 (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 43 h1 (* h2 h2 h2) h3 (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 127 h1 (* h2 h2 h2) h3 (* h5
 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 153 h1 (* h2 h2 h2) h3 (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 82 h1 (* h2 h2 h2) h3 (* h5 h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6
) (* j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 37 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 111 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 139 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2
 j2 j2 j2 j2 j2)) (* 78 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
 j2 j2)) (* 16 h1 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 
h1 (* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
3 h1 (* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3
 h1 (* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* h1 
(* h2 h2 h2) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 
h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 h1 (* 
h2 h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 6 h1 (* 
h2 h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 
h2 h2) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 h1 (* h2 
h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 h1 (* 
h2 h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* 
h2 h2 h2) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2
 h2) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 
h2) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) 
h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2) h4 (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2) h4 (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2) h4 (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2 h2) h4 (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2 h2) h4 (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6
) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) h4 (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2 h2) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 2 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 6 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 6 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 2 h1 (* h2 h2 h2) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (* h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 3 h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))
 (* h1 (* h2 h2 h2) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 72 h1
 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1104 h1 
(* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 4688 h1 (* 
h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 6592 h1 (* h2 h2) 
(* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 3776 h1 (* h2 h2) (* h3 h3 
h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3 h3) (* h4 
h4) h5 (* j2 j2 j2)) (* 72 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2
 j2 j2 j2 j2 j2)) (* 1104 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 
j2 j2 j2 j2)) (* 4688 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 
j2 j2)) (* 6592 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) 
(* 3776 h1 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 768 h1 (* 
h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) (* 108 h1 (* h2 h2) (* h3 h3 
h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1872 h1 (* h2 h2) (* h3 h3 h3 h3
) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 8400 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 
h6 (* j2 j2 j2 j2 j2 j2)) (* 12288 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 
j2 j2 j2 j2)) (* 7296 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 
1536 h1 (* h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 36 h1 (* h2 h2) (* 
h3 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 h1 (* h2 h2) (* h3
 h3 h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 976 h1 (* h2 h2) (* h3 h3 
h3 h3) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 896 h1 (* h2 h2) (* h3 h3 h3 h3) 
(* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 256 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5
) (* j2 j2 j2 j2)) (* 36 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2)) (* 3712 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2)) (* 5696 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 
3520 h1 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 768 h1 (* h2 
h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 36 h1 (* h2 h2) (* h3 h3 h3 h3
) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3 h3) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 3712 h1 (* h2 h2) (* h3 h3 h3 h3) h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 5696 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 
h6) (* j2 j2 j2 j2 j2)) (* 3520 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (- (* 
12 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 196 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 1504 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 
2496 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 1632 h1
 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 384 h1 (* h2 h2) 
(* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 30 h1 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 698 h1 (* h2 h2) (* h3 h3
 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 3476 h1 (* h2 h2) (* h3
 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 5472 h1 (* h2 h2) (* h3
 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 3456 h1 (* h2 h2) (* h3 h3
 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) (- (* 36 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 552 h1 (* h2 h2) (* h3 h3 h3) (* h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4320 h1 (* h2 h2) (* h3 h3 h3) (* h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 7280 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) 
h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 4832 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 
(* j2 j2 j2 j2 j2)) (* 1152 h1 (* h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 
j2 j2)) (* 12 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 356 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 1840 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)
) (* 2848 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 
1760 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 384 h1 (* 
h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 42 h1 (* h2 h2) (* h3 h3
 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 1222 h1 (* h2 h2) (* h3 
h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 6412 h1 (* h2 h2) (* h3 
h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 10368 h1 (* h2 h2) (* h3 h3 
h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 6720 h1 (* h2 h2) (* h3 h3 h3) h4 
(* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1536 h1 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) 
h6 (* j2 j2 j2 j2)) (- (* 36 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (* 516 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 4128 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 7072 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 
j2 j2 j2 j2)) (* 4768 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 
j2)) (* 1152 h1 (* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 36 h1
 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 192 h1 
(* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 208 h1 (* h2
 h2) (* h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 h1 (* h2 h2) (* 
h3 h3 h3) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2) (* h3 h3 h3) 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 320 h1 (* h2 h2) (* h3 h3 h3)
 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1648 h1 (* h2 h2) (* h3 h3 h3) 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2640 h1 (* h2 h2) (* h3 h3 h3) (* 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 1696 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 
h5) h6 (* j2 j2 j2 j2 j2)) (* 384 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* 
j2 j2 j2 j2)) (* 12 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2
 j2 j2 j2 j2 j2)) (* 524 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 2936 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 4896 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2)) (* 3264 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2)) (* 768 h1 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2))
 (- (* 12 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (* 160 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 1312 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))
 (* 2288 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 
1568 h1 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 384 h1 (* 
h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 6 h1 (* h2 h2) (* h3 
h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 50 h1 (* h2 h2) (* h3
 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 168 h1 (* h2 h2) (* h3 
h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 160 h1 (* h2 h2) (* h3 h3) 
(* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2) (* h3 h3) (* h4 h4 
h4 h4) h5 (* j2 j2 j2 j2 j2)) (- (* 3 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 54 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4
) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 513 h1 (* h2 h2) (* h3 h3) (* h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1004 h1 (* h2 h2) (* h3 h3) (* 
h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 740 h1 (* h2 h2) (* h3 h3) (* 
h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) (* h4 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 24 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 200 h1 (* h2 h2) (* h3 h3) (* h4 h4 
h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 672 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4
) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 640 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 
h6 (* j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2
 j2 j2 j2 j2)) (- (* 3 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2 j2))) (* 66 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 545 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1044 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5
 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 756 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 
h5) (* j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2)) (- (* 3 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 174 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 1465 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2900 h1 (* h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2188 h1 (* h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 576 h1 (* h2 h2) (* h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 36 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5 (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 300 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1008 h1 (* h2 h2) (* h3 h3) (* h4 h4)
 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 960 h1 (* h2 h2) (* h3 h3) (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 288 h1 (* h2 h2) (* h3 h3) (* h4 h4) h5 
(* h6 h6) (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 116 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 252 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2
 j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2
 j2)) (* 144 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 1016 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 1976 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2)) (* 1480 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 384 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 3 h1 (* 
h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
186 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 1391 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 2788 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (* 2156 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)
) (* 576 h1 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (- 
(* 24 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)))
 (* 200 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 672 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
640 h1 (* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 h1 
(* h2 h2) (* h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2) 
(* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 116 h1 (* h2 h2)
 (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 252 h1 (* h2 h2) 
(* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3
 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2) (* h3 h3) (* 
h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 78 h1 (* h2 h2) (* h3 h3) (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 471 h1 (* h2 h2) (* h3 h3) (* 
h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 932 h1 (* h2 h2) (* h3 h3) 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 724 h1 (* h2 h2) (* h3 h3) 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) (* 
h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) (* h3 h3) (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 h1 (* h2 h2) (* h3 h3) (* 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 439 h1 (* h2 h2) (* h3 h3
) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 892 h1 (* h2 h2) (* h3 
h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 708 h1 (* h2 h2) (* h3 
h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 h1 (* h2 h2) (* h3 h3) 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (- (* 6 h1 (* h2 h2) (* h3 h3) h5 (* 
h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 50 h1 (* h2 h2) (* h3 h3) h5 
(* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 168 h1 (* h2 h2) (* h3 h3) h5 
(* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 160 h1 (* h2 h2) (* h3 h3) h5 (* h6
 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 48 h1 (* h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6
) (* j2 j2 j2 j2 j2)) (- (* h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 4 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 25 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 32 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 
j2 j2)) (- (* 2 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2))) (* 32 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2
 j2 j2 j2 j2)) (* 90 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2
 j2 j2 j2)) (* 80 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2
 j2)) (* 24 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- 
(* h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)))
 (* 22 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 107 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 132 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 48
 h1 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* h1 (* h2
 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 16 h1 
(* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 45 h1 
(* h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 40 h1 (* 
h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2)
 h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 102 h1 (* h2 h2) h3 (* 
h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 274 h1 (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 244 h1 (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 72 h1 (* h2 h2) h3 (* h4 
h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) h3 (* h4 h4) (* h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 42 h1 (* h2 h2) h3 (* h4 h4)
 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 171 h1 (* h2 h2) h3 (* 
h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 204 h1 (* h2 h2) h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 72 h1 (* h2 h2) h3 
(* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2) h3 h4 (* h5
 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 38 h1 (* h2 h2) h3 h4 (* h5 
h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 97 h1 (* h2 h2) h3 h4 (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 h1 (* h2 h2) h3 h4 (* h5 h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 24 h1 (* h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2)) (* 6 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 108 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2)) (* 278 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 248 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2
 j2 j2 j2 j2)) (* 72 h1 (* h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2
 j2)) (* 5 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 34 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 121 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2)) (* 140 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2))
 (* 48 h1 (* h2 h2) h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2 h1 
(* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 22 
h1 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 52 
h1 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 h1 
(* h2 h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 
h2) h3 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2) h3 (* 
h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 38 h1 (* h2 h2) h3 
(* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 94 h1 (* h2 h2) h3 
(* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 h1 (* h2 h2) h3 (* 
h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 h1 (* h2 h2) h3 (* h5 h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) h3 (* h5 h5) (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 10 h1 (* h2 h2) h3 (* h5 h5) (* h6 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 h1 (* h2 h2) h3 (* h5 h5) (* h6 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 36 h1 (* h2 h2) h3 (* h5 h5) (* h6 h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 12 h1 (* h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6)
 (* j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 3 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 6 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h2 h2) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 6 h1 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 3 h1 (* h2 h2) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 2 h1 (* h2 h2) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* h1 (* h2 h2) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* h1 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 2 h1 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* h1 (* h2 h2) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (- (* 6 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2
 j2))) (- (* 28 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2
))) (* 128 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 
864 h1 h2 (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 1280 h1 h2 
(* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3 h3)
 (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 6 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 172 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1264 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2)) (* 3456 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* 
j2 j2 j2 j2 j2 j2)) (* 3328 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2)) (* 1024 h1 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2)) 
(- (* 18 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 84 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 
384 h1 h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2592 h1 
h2 (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 3840 h1 h2 (* h3 h3
 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 1536 h1 h2 (* h3 h3 h3 h3) (* h4 
h4) h5 h6 (* j2 j2 j2 j2)) (* 48 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 464 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2)) (* 1440 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(* 1536 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 512 h1 h2 
(* h3 h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2)) (* 6 h1 h2 (* h3 h3 h3 h3) h4 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 268 h1 h2 (* h3 h3 h3 h3) h4 (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2192 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5
) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 6336 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2)) (* 6400 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2
 j2)) (* 2048 h1 h2 (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (- (* 18 h1
 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 84 h1
 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 384 h1 h2 
(* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2592 h1 h2 (* h3 h3
 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 3840 h1 h2 (* h3 h3 h3 h3) h4 
h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 1536 h1 h2 (* h3 h3 h3 h3) h4 h5 (* h6 h6) 
(* j2 j2 j2 j2)) (* 48 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 464 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)
) (* 1440 h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 1536 
h1 h2 (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 
h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 96 h1 h2 (* h3 h3 h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 928 h1 h2 (* h3 h3 h3 h3) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2880 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 3072 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2)) (* 1024 h1 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2)) (- (* 6 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2))) (- (* 28 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
))) (* 128 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
864 h1 h2 (* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1280 h1 h2 
(* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3 h3)
 h5 (* h6 h6 h6) (* j2 j2 j2 j2)) (- (* 6 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 
(* j2 j2 j2 j2 j2 j2 j2 j2))) (* 144 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2
 j2 j2 j2 j2 j2 j2)) (* 288 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2
 j2 j2)) (* 128 h1 h2 (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (- (* 
3 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 9 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 318 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 1256 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
1408 h1 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 512 h1 
h2 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* 24 h1 h2 (* h3 
h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 16 h1 h2 (* h3 
h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 576 h1 h2 (* h3 h3 h3
) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1152 h1 h2 (* h3 h3 h3) (* h4 
h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3) (* h4 h4 h4) h5 h6
 (* j2 j2 j2 j2 j2)) (- (* 3 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2))) (* 15 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* 370 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 1384 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 1472 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 
j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2
 j2)) (- (* 5 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2 j2))) (* 55 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 986 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 3704 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 4224 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2))
 (* 1536 h1 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* 36 
h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 24 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 864 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
1728 h1 h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 768 h1 
h2 (* h3 h3 h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 48 h1 h2 (* h3 h3 
h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 272 h1 h2 (* h3 h3 h3) h4
 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 352 h1 h2 (* h3 h3 h3) h4 (* h5 h5
 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* 
j2 j2 j2 j2 j2)) (- (* 2 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2 j2 j2 j2))) (* 58 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2 j2 j2)) (* 772 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2
 j2 j2)) (* 2704 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2))
 (* 2944 h1 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 1024 h1
 h2 (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* h1 h2 (* h3 h3 h3)
 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 83 h1 h2 (* h3 h3
 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 1018 h1 h2 (* h3 
h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 3640 h1 h2 (* h3 
h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4224 h1 h2 (* h3 h3 
h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1536 h1 h2 (* h3 h3 h3) h4 
(* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (- (* 24 h1 h2 (* h3 h3 h3) h4 h5 (* h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 16 h1 h2 (* h3 h3 h3) h4 h5 (* h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 576 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 h6
) (* j2 j2 j2 j2 j2 j2 j2)) (* 1152 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)
) (* 48 h1 h2 (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
272 h1 h2 (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 352 h1 h2
 (* h3 h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3 h3
) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 43 h1 h2 (* h3 h3 h3) (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 402 h1 h2 (* h3 h3 h3) (* h5 h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1320 h1 h2 (* h3 h3 h3) (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1472 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 512 h1 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2)) (* h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 37 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2 j2 j2)) (* 350 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2)) (* 1192 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 1408 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 512 h1 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (- 
(* 6 h1 h2 (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 4 h1 h2 (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 144 
h1 h2 (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 288 h1 h2 (* 
h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3 h3) h5
 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2)) (- (* 2 h1 h2 (* h3 h3) (* h4 h4 h4 h4) (* 
h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 2 h1 h2 (* h3 h3) (* h4 h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 68 h1 h2 (* h3 h3) (* h4 h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3) (* h4 h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 h2 (* h3 h3) (* h4 h4 h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2)) (- (* 4 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 22 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 220 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 304 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2)) (- (* 5 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2 j2 j2 j2))) (* 23 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 288 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 528 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2
 j2 j2 j2 j2 j2)) (* 256 h1 h2 (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 
j2 j2 j2)) (- (* 2 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2 j2))) (* 8 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 96 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 144 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2
 j2 j2)) (* 64 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) 
(- (* 6 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (* 90 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2
 j2)) (* 664 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 928 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 384 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 3 
h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))
) (* 57 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2)) (* 456 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 816 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 384 h1 h2 (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (- (* h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 
j2))) (* 31 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 208 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
304 h1 h2 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 128 h1 h2
 (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 114 h1 h2 (* h3 h3) 
h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 668 h1 h2 (* h3 h3)
 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 944 h1 h2 (* h3 h3) 
h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 384 h1 h2 (* h3 h3) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* h1 h2 (* h3 h3) h4 (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 53 h1 h2 (* h3 h3) h4 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 320 h1 h2 (* h3 h3) h4 (* h5
 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 560 h1 h2 (* h3 h3) h4 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 256 h1 h2 (* h3 h3) h4 (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6)
 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 23 h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 112 h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 160 h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 2 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 46 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 224 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 320 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 128 h1 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2)) (* h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 17 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 84 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2 j2)) (* 144 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2)) (* 64 h1 h2 (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (- 
(* h1 h2 h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* h1 h2 h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 10 
h1 h2 h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 h2 h3
 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (- (* h1 h2 h3 (* h4 h4 
h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* h1 h2 h3 (* h4 h4 
h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 10 h1 h2 h3 (* h4 h4 h4)
 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 h2 h3 (* h4 h4 h4) (* h5 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (- (* 2 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 6 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 h1 h2 h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2)) (- (* h1 h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2 j2))) (* 7 h1 h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 38 h1 h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 24 h1 h2 h3 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2))
 (* 24 h1 h2 h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 84 h1 h2 h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 48 h1 h2 h3 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* h1 
h2 h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 17 h1 
h2 h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 46 h1 h2 
h3 h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 24 h1 h2 h3 h4 
(* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2 h1 h2 h3 h4 (* h5 h5 h5
) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 26 h1 h2 h3 h4 (* h5 h5 h5)
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 h2 h3 h4 (* h5 h5 h5) (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 h1 h2 h3 h4 (* h5 h5 h5) (* h6 h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 9 h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2)) (* 18 h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 8 h1 h2 h3 (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
)) (* h1 h2 h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 9 h1 h2 h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 18
 h1 h2 h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 8 h1 h2 
h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3 h3 
h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 h1 (* h3
 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 44 h1 
(* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 16 h1
 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2))) (* 192 h1 (* 
h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 h1 (* h3 h3 h3
 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3 h3 h3) (* h4 
h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 12 h1 (* h3 h3 h3 h3)
 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 44 h1 (* h3 h3 h3
 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 16 h1 (* h3 h3 
h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2))) (* 192 h1 (* h3 h3 h3 
h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 h1 (* h3 h3 h3 h3) (* 
h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (- (* 2 h1 (* h3 h3 h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 20 h1 (* h3 h3 h3 h3) (* h4 
h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 36 h1 (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2))) (* 208 h1 (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 832 h1 (* h3 h3 h3 h3) (* h4
 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 768 h1 (* h3 h3 h3 h3) (* h4 h4) (* 
h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 8 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 224 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 640 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)
) (* 512 h1 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (- (* h1 (* 
h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 
h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 60
 h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 464 
h1 (* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1088 h1 
(* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 768 h1 (* h3 h3
 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 4 h1 (* h3 h3 h3 h3) (* h5
 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 52 h1 (* h3 h3 h3 h3) (* h5
 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 h1 (* h3 h3 h3 h3) (* h5 
h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 448 h1 (* h3 h3 h3 h3) (* h5 h5 h5
) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 256 h1 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2)) (* 4 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 52 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 240 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2)) (* 448 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 256 h1 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (- 
(* h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2)))
 (- (* 8 h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (- (* 12 h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2))) (* 32 h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 64 h1 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 2 
h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 16 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (- (* 24 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2))) (* 64 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 128 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 3 
h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 20 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 192 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 256 h1 (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* h1
 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- 
(* 8 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 12 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)))
 (* 32 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 64
 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (- (* 4 h1 (* 
h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 24
 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 
24 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 320
 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 384 h1 
(* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 3 h1 (* h3 
h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 
12 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
) (* 72 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
)) (* 384 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)
) (* 384 h1 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(- (* h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(- (* 4 h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 24 h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 
h1 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 128 h1 (* h3 
h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (- (* 2 h1 (* h3 h3 h3) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 120 h1 (* h3 h3 h3
) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 448 h1 (* h3 h3 h3) 
h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 384 h1 (* h3 h3 h3) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3 h3) h4 (* h5 h5)
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 4 h1 (* h3 h3 h3) h4 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 h1 (* h3 h3 h3) h4 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 320 h1 (* h3 h3 h3) h4 (* h5 h5
) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 256 h1 (* h3 h3 h3) h4 (* h5 h5) (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 36 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2)) (* 8 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 72 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 192 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2)) (* 128 h1 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(* 4 h1 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 36 h1 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
96 h1 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 
(* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3)
 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 h1 (* 
h3 h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 4 h1 (* 
h3 h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h3 
h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (- (* h1 (* h3 h3) 
(* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 h1 (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 4 h1 (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h3 
h3) (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (- (* 3 h1 (* h3 h3) 
(* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 8 h1 (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 32 h1 
(* h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 h1 (* 
h3 h3) (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (- (* 2 h1 (* h3 
h3) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (- (* 4 h1 
(* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 28 h1
 (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 h1 
(* h3 h3) (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (- (* 3 h1 (* 
h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) (* 
72 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 96 h1 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(- (* h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2
))) (* 4 h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2
)) (* 44 h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 48 h1 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (- (* 
h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2 j2))) 
(* 8 h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 64 h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
64 h1 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 
(* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 20 h1 
(* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h1 (* 
h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 h1 (* h3 h3) 
(* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 20 h1 (* h3 h3) 
(* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h1 (* h3 h3) (* 
h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 216 (* h2 h2 h2 h2) (* h3
 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 648 (* h2 h2 h2 h2) (* h3 h3 
h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 720 (* h2 h2 h2 h2) (* h3 h3 h3) (* 
h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 352 (* h2 h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 
(* j2 j2 j2 j2)) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) 
(* 216 (* h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
648 (* h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 720 (* 
h2 h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2)) (* 352 (* h2 h2 h2 h2
) (* h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2)) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) 
h4 (* h5 h5) (* j2 j2 j2)) (* 432 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2
 j2 j2 j2 j2 j2)) (* 1296 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 
j2 j2)) (* 1440 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 704
 (* h2 h2 h2 h2) (* h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 128 (* h2 h2 h2 h2) 
(* h3 h3 h3) h4 h5 h6 (* j2 j2 j2)) (* 216 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5
) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 648 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 
(* j2 j2 j2 j2 j2 j2)) (* 720 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2
 j2 j2 j2)) (* 352 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) 
(* 64 (* h2 h2 h2 h2) (* h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 216 (* h2 h2 h2
 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 648 (* h2 h2 h2 h2) 
(* h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 720 (* h2 h2 h2 h2) (* h3 h3
 h3) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 352 (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* 
h6 h6) (* j2 j2 j2 j2)) (* 64 (* h2 h2 h2 h2) (* h3 h3 h3) h5 (* h6 h6) (* j2 j2
 j2)) (* 36 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2
)) (* 120 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 148 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 80 
(* h2 h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2 
h2) (* h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 72 (* h2 h2 h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3
) (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 296 (* h2 h2 h2 h2) (* h3 h3)
 (* h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 160 (* h2 h2 h2 h2) (* h3 h3) (* 
h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 32 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2)) (* 108 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 360 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2
 j2 j2 j2 j2 j2 j2)) (* 444 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 
j2 j2 j2 j2)) (* 240 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2
)) (* 48 (* h2 h2 h2 h2) (* h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 36 (* h2 
h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 120 (* h2 h2
 h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 148 (* h2 h2 h2 
h2) (* h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3
 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2 h2) (* h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2 j2)) (* 144 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 480 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2)) (* 592 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 
j2 j2 j2 j2)) (* 320 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2
)) (* 64 (* h2 h2 h2 h2) (* h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 108 (* h2
 h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 360 (* h2 
h2 h2 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 444 (* h2 h2 h2
 h2) (* h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 240 (* h2 h2 h2 h2) (* 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 48 (* h2 h2 h2 h2) (* h3 h3) h4 h5
 (* h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2)) (* 148 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 
j2 j2 j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)
) (* 16 (* h2 h2 h2 h2) (* h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 72 (* h2 
h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 240 (* 
h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 296 (* 
h2 h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 160 (* h2 
h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 32 (* h2 h2 h2 h2
) (* h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2 h2 h2) (* h3 h3)
 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 120 (* h2 h2 h2 h2) (* h3 h3) 
h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 148 (* h2 h2 h2 h2) (* h3 h3) h5 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 80 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 
h6) (* j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6) (* j2 j2
 j2 j2)) (* 6 (* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 22 (* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 30 (* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 18 (* h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* 
h2 h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 6 (* h2 h2 h2 h2) 
h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 22 (* h2 h2 h2 h2) 
h3 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h2 h2 h2 h2) h3 
(* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h2 h2 h2 h2) h3 (* h4 
h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 
h5 h5) (* j2 j2 j2 j2 j2)) (* 18 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 90 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2
 j2 j2 j2 j2 j2)) (* 54 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2
 j2 j2)) (* 12 (* h2 h2 h2 h2) h3 (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 
12 (* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 44 
(* h2 h2 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 (* h2 
h2 h2 h2) h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 36 (* h2 h2 h2 h2) 
h3 h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 8 (* h2 h2 h2 h2) h3 h4 (* h5 h5
 h5) h6 (* j2 j2 j2 j2 j2)) (* 18 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 66 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 90 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2 j2)) (* 54 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2)) (* 12 (* h2 h2 h2 h2) h3 h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) 
(* 6 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 22 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
30 (* h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* 
h2 h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2 
h2) h3 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 6 (* h2 h2 h2 h2) h3 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 22 (* h2 h2 h2 h2) h3 (* h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h2 h2 h2 h2) h3 (* h5 h5)
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6
 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2
 j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 
j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2 j2)) 
(* 1216 (* h2 h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2 j2)) (* 256 (* h2 
h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 (* j2 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3 
h3 h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 
h3) h4 (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) h4 
(* h5 h5) (* j2 j2 j2 j2 j2)) (* 1216 (* h2 h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) 
(* j2 j2 j2 j2)) (* 256 (* h2 h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) (* j2 j2 j2)) 
(* 432 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2592 
(* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 4032 (* h2 h2 h2
) (* h3 h3 h3 h3) h4 h5 h6 (* j2 j2 j2 j2 j2)) (* 2432 (* h2 h2 h2) (* h3 h3 h3 
h3) h4 h5 h6 (* j2 j2 j2 j2)) (* 512 (* h2 h2 h2) (* h3 h3 h3 h3) h4 h5 h6 (* j2
 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 
j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 1216 
(* h2 h2 h2) (* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2 j2)) (* 256 (* h2 h2 h2) 
(* h3 h3 h3 h3) (* h5 h5) h6 (* j2 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3 h3 h3) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1296 (* h2 h2 h2) (* h3 h3 h3 h3) h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2016 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 
h6) (* j2 j2 j2 j2 j2)) (* 1216 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 
j2 j2 j2)) (* 256 (* h2 h2 h2) (* h3 h3 h3 h3) h5 (* h6 h6) (* j2 j2 j2)) (* 72 
(* h2 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 456 
(* h2 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 800 (* h2
 h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 544 (* h2 h2 h2) 
(* h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 128 (* h2 h2 h2) (* h3 h3 h3
) (* h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 144 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 912 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4
) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1600 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4
) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 1088 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2)) (* 256 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) (* h5 
h5) (* j2 j2 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 1368 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 
j2 j2 j2 j2 j2)) (* 2400 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 
j2 j2 j2)) (* 1632 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2))
 (* 384 (* h2 h2 h2) (* h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 72 (* h2 
h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 456 (* h2 h2
 h2) (* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 800 (* h2 h2 h2) 
(* h3 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 544 (* h2 h2 h2) (* h3 h3
 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 128 (* h2 h2 h2) (* h3 h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2 j2)) (* 288 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 1824 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2)) (* 3200 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 
j2 j2 j2 j2 j2)) (* 2176 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 
j2 j2)) (* 512 (* h2 h2 h2) (* h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 216
 (* h2 h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1368 
(* h2 h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2400 (* 
h2 h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 1632 (* h2 h2 
h2) (* h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 384 (* h2 h2 h2) (* h3 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2)) (* 72 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5
 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 456 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 800 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) 
h6 (* j2 j2 j2 j2 j2 j2)) (* 544 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2)) (* 128 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2))
 (* 144 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
)) (* 912 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)
) (* 1600 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(* 1088 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 256
 (* h2 h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 72 (* h2 h2 
h2) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 456 (* h2 h2 h2
) (* h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 800 (* h2 h2 h2) (* 
h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 544 (* h2 h2 h2) (* h3 h3 h3
) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 128 (* h2 h2 h2) (* h3 h3 h3) h5 (* h6 
h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2
 j2 j2 j2 j2 j2 j2)) (* 84 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2
 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2
 j2)) (* 16 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 18 
(* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 156 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 318 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2))
 (* 244 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 
64 (* h2 h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 144 (* 
h2 h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2
 h2 h2) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2 
h2) (* h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3
 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 18 (* h2 h2 h2) (* h3 h3) (* h4 
h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 156 (* h2 h2 h2) (* h3 h3) 
(* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 318 (* h2 h2 h2) (* h3 h3
) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 244 (* h2 h2 h2) (* h3 h3)
 (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) (* 
h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 54 (* h2 h2 h2) (* h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 468 (* h2 h2 h2) (* h3 h3) (* h4
 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 954 (* h2 h2 h2) (* h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 732 (* h2 h2 h2) (* h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 192 (* h2 h2 h2) (* h3 h3) (* h4 
h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 216 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 504 (* h2 h2 h2) (* h3 h3) (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 384 (* h2 h2 h2) (* h3 h3) (* h4 h4) 
h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 96 (* h2 h2 h2) (* h3 h3) (* h4 h4) h5 
(* h6 h6) (* j2 j2 j2 j2 j2)) (* 36 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 
j2 j2 j2 j2)) (* 16 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2)
) (* 36 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 312 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 636 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 
488 (* h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 128 (* 
h2 h2 h2) (* h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 54 (* h2 h2 h2) 
(* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 468 (* h2 h2
 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 954 (* h2 
h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 732 (* h2 
h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 192 (* h2 h2 
h2) (* h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 144 (* h2 h2 h2) (* 
h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2 h2 h2) (* h3
 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2 h2) (* h3 h3) 
h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) h4 h5 (* 
h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 36 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 
j2 j2 j2 j2)) (* 16 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2)
) (* 18 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
 j2)) (* 156 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 318 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 244 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 64 (* h2 h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 
18 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 156 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2
)) (* 318 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)
) (* 244 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(* 64 (* h2 h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 36 
(* h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* 
h2 h2 h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 
h2) (* h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2) (* h3
 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 6 (* h2 h2 h2) h3 (* h4 h4 h4 h4)
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2) h3 (* h4 h4 h4 h4)
 (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h2 h2 h2) h3 (* h4 h4 h4 h4) 
(* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2) h3 (* h4 h4 h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2)) (* 12 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 28 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2
 j2 j2 j2 j2 j2)) (* 8 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 
j2 j2)) (* 24 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 64 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 56 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) 
(* 16 (* h2 h2 h2) h3 (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 6 (* 
h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* 
h2 h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h2 
h2 h2) h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2) 
h3 (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 36 (* h2 h2 h2) h3 (* h4 
h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 (* h2 h2 h2) h3 (* h4 
h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h2 h2 h2) h3 (* h4 h4)
 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h2 h2 h2) h3 (* h4 h4) (* h5
 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 36 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) (* 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5
) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h2 h2 h2) h3 (* h4 h4) (* h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 12 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 28 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 8 (* h2 h2 h2) h3 h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 36 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 96 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
84 (* h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* 
h2 h2 h2) h3 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h2 h2 h2)
 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2 h2)
 h3 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 56 (* h2 h2 h2) h3
 h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2) h3 h4 
(* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 6 (* h2 h2 h2) h3 (* h5 h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h2 h2 h2) h3 (* h5 h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 14 (* h2 h2 h2) h3 (* h5 h5 h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2 h2) h3 (* h5 h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2)) (* 12 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2 j2)) (* 32 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 28 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2
 j2 j2 j2 j2 j2)) (* 8 (* h2 h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 6 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 16 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 14 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 4 (* h2 h2 h2) h3 (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 36 (* 
h2 h2) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2
 h2) (* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h2 h2) 
(* h3 h3 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2 j2)) (* 896 (* h2 h2) (* h3 h3
 h3 h3) (* h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3 h3) (* 
h4 h4 h4) h5 (* j2 j2 j2 j2)) (* 72 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 672 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 1952 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 
h5) (* j2 j2 j2 j2 j2 j2)) (* 1792 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5)
 (* j2 j2 j2 j2 j2)) (* 512 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) (* h5 h5) (* j2 
j2 j2 j2)) (* 108 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2
 j2 j2)) (* 1008 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 
j2)) (* 2928 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) 
(* 2688 (* h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 768 (* 
h2 h2) (* h3 h3 h3 h3) (* h4 h4) h5 h6 (* j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3
 h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2 h2) (* h3 h3 
h3 h3) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h2 h2) (* h3 h3 h3 h3
) h4 (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 h3) h4 (* 
h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5 h5) 
(* j2 j2 j2 j2)) (* 144 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 1344 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 3904 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2 
j2)) (* 3584 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 
1024 (* h2 h2) (* h3 h3 h3 h3) h4 (* h5 h5) h6 (* j2 j2 j2 j2)) (* 108 (* h2 h2)
 (* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1008 (* h2 h2) 
(* h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 2928 (* h2 h2) (* 
h3 h3 h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 2688 (* h2 h2) (* h3 h3 
h3 h3) h4 h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 768 (* h2 h2) (* h3 h3 h3 h3) h4 
h5 (* h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 
(* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2
 j2 j2 j2 j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5 h5) h6 (* j2 j2 j2 j2)) (* 72 
(* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
672 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 
1952 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 
1792 (* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 512 
(* h2 h2) (* h3 h3 h3 h3) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2) 
(* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 (* h2 h2) 
(* h3 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 976 (* h2 h2) (* h3
 h3 h3 h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 896 (* h2 h2) (* h3 h3 h3 
h3) h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3 h3) h5 (* 
h6 h6 h6) (* j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2
 j2 j2 j2 j2 j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 
j2 j2 j2 j2 j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2
 j2 j2 j2)) (* 64 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4 h4) h5 (* j2 j2 j2 j2 j2)) 
(* 18 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 732 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 
j2 j2)) (* 784 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 
j2)) (* 256 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) 
(* 144 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 768 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 
832 (* h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2 j2)) (* 256 (* 
h2 h2) (* h3 h3 h3) (* h4 h4 h4) h5 h6 (* j2 j2 j2 j2 j2)) (* 18 (* h2 h2) (* h3
 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 210 (* h2 h2) 
(* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 732 (* h2 h2
) (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 784 (* h2 h2)
 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* 
h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 54 (* h2 h2) (* h3 h3 h3
) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 630 (* h2 h2) (* h3 
h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 2196 (* h2 h2) (* 
h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 2352 (* h2 h2) (* 
h3 h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 768 (* h2 h2) (* h3 
h3 h3) (* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 216 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1152 (* h2 h2) (* h3 h3 
h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1248 (* h2 h2) (* h3 h3 
h3) (* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 384 (* h2 h2) (* h3 h3 h3) 
(* h4 h4) h5 (* h6 h6) (* j2 j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3) h4 (* h5
 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) h4 (* h5 
h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5
 h5) (* j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5 h5) (* 
j2 j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 420 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 1464 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 1568 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2)) (* 512 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 54 
(* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 630 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 2196 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)
) (* 2352 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) 
(* 768 (* h2 h2) (* h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 144 
(* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 768 
(* h2 h2) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 832 (* h2
 h2) (* h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* 
h3 h3 h3) h4 h5 (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) 
(* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) (* h5
 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 64 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5 h5
) h6 (* j2 j2 j2 j2 j2)) (* 18 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6
) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 732 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 784 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 
h6) (* j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3) (* h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2)) (* 18 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2
 j2 j2 j2 j2 j2 j2 j2)) (* 210 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2)) (* 732 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 784 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2)) (* 256 (* h2 h2) (* h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2)) (* 36 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2
 j2 j2)) (* 208 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2))
 (* 64 (* h2 h2) (* h3 h3 h3) h5 (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 12 (* h2
 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 68 
(* h2 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 88
 (* h2 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 32 
(* h2 h2) (* h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24 (* h2
 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 136 
(* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
176 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 
64 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 48 
(* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 272 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 352 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2))
 (* 128 (* h2 h2) (* h3 h3) (* h4 h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 
12 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2))
 (* 68 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 88 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 32 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 72
 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 408 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)
) (* 528 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2))
 (* 192 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 
72 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 408 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 528 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 192 (* h2 h2) (* h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 
j2 j2 j2 j2)) (* 24 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 136 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2
 j2 j2 j2)) (* 176 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 
j2 j2)) (* 64 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) 
(* 72 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 408 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 
j2 j2)) (* 528 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2)) (* 192 (* h2 h2) (* h3 h3) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2)) (* 48 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2
 j2 j2)) (* 272 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 352 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 128 (* h2 h2) (* h3 h3) h4 (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 12 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 68 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 88 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2
 j2 j2 j2)) (* 32 (* h2 h2) (* h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 24 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2
 j2 j2 j2)) (* 136 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2)) (* 176 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 64 (* h2 h2) (* h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
 j2 j2)) (* 12 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 68 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 88 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 32 (* h2 h2) (* h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2
 j2 j2)) (* 6 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 10 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 
j2)) (* 4 (* h2 h2) h3 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) 
(* 6 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 10 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 
4 (* h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* 
h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 40 (* 
h2 h2) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h2 
h2) h3 (* h4 h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h2 h2) h3
 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h2 h2) h3
 (* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 36 (* h2 h2) h3 (* h4 
h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 60 (* h2 h2) h3 
(* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h2 h2) h3
 (* h4 h4) (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 (* h2 h2) h3 
h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 30 (* h2 h2) h3 
h4 (* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 12 (* h2 h2) h3 h4 
(* h5 h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 (* h2 h2) h3 h4 (* h5 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 40 (* h2 h2) h3 h4 (* h5 
h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 (* h2 h2) h3 h4 (* h5 h5 
h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h2 h2) h3 (* h5 h5 h5 h5) (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 10 (* h2 h2) h3 (* h5 h5 h5 h5) (* 
h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2) h3 (* h5 h5 h5 h5) (* h6 
h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 6 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 10 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 4 (* h2 h2) h3 (* h5 h5 h5) (* h6 h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 76 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 336 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2
 j2 j2 j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 
j2 j2)) (* 256 h2 (* h3 h3 h3 h3) (* h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2)) (* 
6 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 
76 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336
 h2 (* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 576 h2 
(* h3 h3 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 256 h2 (* h3 h3
 h3 h3) (* h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2)) (* 18 h2 (* h3 h3 h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 228 h2 (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1008 h2 (* h3 h3 h3 h3) 
(* h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 1728 h2 (* h3 h3 h3 h3) (* 
h4 h4) (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 768 h2 (* h3 h3 h3 h3) (* h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 12 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 152 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 
j2 j2 j2 j2 j2 j2 j2)) (* 672 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 
j2 j2 j2 j2)) (* 1152 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2
)) (* 512 h2 (* h3 h3 h3 h3) h4 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2)) (* 18 h2 (* 
h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 228 h2 
(* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 1008 h2 
(* h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 1728 h2 (* 
h3 h3 h3 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 768 h2 (* h3 h3 h3
 h3) h4 (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3 h3 h3) (* h5 h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 76 h2 (* h3 h3 h3 h3) (* h5 h5
 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 336 h2 (* h3 h3 h3 h3) (* h5 h5 
h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 256 h2 (* h3 h3 h3 h3) (* h5 h5 h5) (* h6 h6
) (* j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2 j2)) (* 76 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2
 j2 j2 j2 j2 j2)) (* 336 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 576 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2
 j2)) (* 256 h2 (* h3 h3 h3 h3) (* h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2)) (* 6 
h2 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 52 
h2 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 h2
 (* h3 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 64 h2 (* h3
 h3 h3) (* h4 h4 h4 h4) (* h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 12 h2 (* h3 h3 h3) 
(* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 104 h2 (* h3 h3 h3)
 (* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 256 h2 (* h3 h3 h3) 
(* h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 128 h2 (* h3 h3 h3) (* h4
 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2)) (* 24 h2 (* h3 h3 h3) (* h4 h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 208 h2 (* h3 h3 h3) (* h4 h4 h4)
 (* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 512 h2 (* h3 h3 h3) (* h4 h4 h4) 
(* h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 256 h2 (* h3 h3 h3) (* h4 h4 h4) (* h5
 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 64 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5 h5) (* j2 
j2 j2 j2 j2 j2)) (* 36 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 312 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 768 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 
j2 j2 j2)) (* 384 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2
)) (* 36 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 
j2 j2)) (* 312 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 768 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 384 h2 (* h3 h3 h3) (* h4 h4) (* h5 h5) (* h6 h6) (* j2 j2 j2 j2 
j2 j2)) (* 12 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 
j2)) (* 104 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 256 h2 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 128 h2
 (* h3 h3 h3) h4 (* h5 h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2)) (* 36 h2 (* h3 h3 h3)
 h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 312 h2 (* h3 h3 h3
) h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 768 h2 (* h3 h3 h3) 
h4 (* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 384 h2 (* h3 h3 h3) h4 
(* h5 h5 h5) (* h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 24 h2 (* h3 h3 h3) h4 (* h5 h5)
 (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 208 h2 (* h3 h3 h3) h4 (* h5 h5
) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 512 h2 (* h3 h3 h3) h4 (* h5 h5) 
(* h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 256 h2 (* h3 h3 h3) h4 (* h5 h5) (* h6
 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 52 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 128 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2)) (* 64 h2 (* h3 h3 h3) (* h5 h5 h5 h5) (* h6 h6) (* j2 
j2 j2 j2 j2 j2)) (* 12 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2 j2 j2)) (* 104 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2
 j2 j2 j2 j2)) (* 256 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 
j2 j2 j2)) (* 128 h2 (* h3 h3 h3) (* h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 j2 j2 j2
)) (* 6 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)
) (* 52 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) 
(* 128 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 64
 h2 (* h3 h3 h3) (* h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 
h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 h2 (* h3 
h3) (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h2 (* h3 h3)
 (* h4 h4 h4 h4) (* h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3) (* h4 
h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 28 h2 (* h3 h3) (* h4 
h4 h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 16 h2 (* h3 h3) (* h4 h4 
h4) (* h5 h5 h5 h5) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 h2 (* h3 h3) (* h4 h4 h4) 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 112 h2 (* h3 h3) (* h4 h4 h4)
 (* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 h2 (* h3 h3) (* h4 h4 h4) 
(* h5 h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 18 h2 (* h3 h3) (* h4 h4) (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 h2 (* h3 h3) (* h4 h4) (* h5 h5 
h5 h5) h6 (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5 
h5) h6 (* j2 j2 j2 j2 j2 j2 j2)) (* 36 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 168 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5) 
(* h6 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 96 h2 (* h3 h3) (* h4 h4) (* h5 h5 h5)
 (* h6 h6) (* j2 j2 j2 j2 j2 j2 j2)) (* 18 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 84 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6
 h6) (* j2 j2 j2 j2 j2 j2 j2 j2)) (* 48 h2 (* h3 h3) h4 (* h5 h5 h5 h5) (* h6 h6
) (* j2 j2 j2 j2 j2 j2 j2)) (* 24 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2 j2 j2)) (* 112 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) 
(* j2 j2 j2 j2 j2 j2 j2 j2)) (* 64 h2 (* h3 h3) h4 (* h5 h5 h5) (* h6 h6 h6) (* 
j2 j2 j2 j2 j2 j2 j2)) (* 6 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2 j2)) (* 28 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 
j2 j2 j2 j2 j2 j2)) (* 16 h2 (* h3 h3) (* h5 h5 h5 h5) (* h6 h6 h6) (* j2 j2 j2 
j2 j2 j2 j2)) (* 6 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2 j2)) (* 28 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 
j2 j2 j2)) (* 16 h2 (* h3 h3) (* h5 h5 h5) (* h6 h6 h6 h6) (* j2 j2 j2 j2 j2 j2 
j2))) 0)))
(check-sat)
(exit)
