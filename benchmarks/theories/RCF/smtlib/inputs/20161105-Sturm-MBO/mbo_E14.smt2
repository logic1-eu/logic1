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
(set-info :instance |E14|)
(set-info :category "industrial")
(set-info :status sat)
(declare-const h6 Real)
(declare-const h5 Real)
(declare-const h4 Real)
(declare-const h3 Real)
(declare-const h2 Real)
(declare-const h1 Real)
(assert (and (> h1 0) (> h2 0) (> h3 0) (> h4 0) (> h5 0) (> h6 0) (= (+ (- (* 2
 (* h1 h1 h1) (* h2 h2) h3)) (* 2 (* h1 h1 h1) (* h2 h2) h5) (- (* 8 (* h1 h1 h1
) h2 (* h3 h3))) (* 4 (* h1 h1 h1) h2 h3 h4) (* 10 (* h1 h1 h1) h2 h3 h5) (* 10 
(* h1 h1 h1) h2 h3 h6) (* 2 (* h1 h1 h1) h2 h4 h5) (* 2 (* h1 h1 h1) h2 (* h5 h5
)) (* 24 (* h1 h1 h1) (* h3 h3) h4) (* 16 (* h1 h1 h1) (* h3 h3) h5) (* 16 (* h1
 h1 h1) (* h3 h3) h6) (* 6 (* h1 h1 h1) h3 (* h4 h4)) (* 6 (* h1 h1 h1) h3 h4 h5
) (* 10 (* h1 h1 h1) h3 h4 h6) (* 4 (* h1 h1 h1) h3 (* h5 h5)) (* 8 (* h1 h1 h1)
 h3 h5 h6) (* 4 (* h1 h1 h1) h3 (* h6 h6)) (* 2 (* h1 h1 h1) h4 h5 h6) (* 2 (* 
h1 h1 h1) (* h5 h5) h6) (* 2 (* h1 h1 h1) h5 (* h6 h6)) (- (* (* h1 h1) (* h2 h2
 h2) h3)) (* (* h1 h1) (* h2 h2 h2) h5) (- (* 9 (* h1 h1) (* h2 h2) (* h3 h3))) 
(* (* h1 h1) (* h2 h2) h3 h4) (* 10 (* h1 h1) (* h2 h2) h3 h5) (* 8 (* h1 h1) 
(* h2 h2) h3 h6) (* 6 (* h1 h1) (* h2 h2) h4 h5) (* 3 (* h1 h1) (* h2 h2) (* h5 
h5)) (* 3 (* h1 h1) (* h2 h2) h5 h6) (- (* 8 (* h1 h1) h2 (* h3 h3 h3))) (* 28 
(* h1 h1) h2 (* h3 h3) h4) (* 26 (* h1 h1) h2 (* h3 h3) h5) (* 34 (* h1 h1) h2 
(* h3 h3) h6) (* 13 (* h1 h1) h2 h3 (* h4 h4)) (* 28 (* h1 h1) h2 h3 h4 h5) (* 
30 (* h1 h1) h2 h3 h4 h6) (* 15 (* h1 h1) h2 h3 (* h5 h5)) (* 32 (* h1 h1) h2 h3
 h5 h6) (* 17 (* h1 h1) h2 h3 (* h6 h6)) (* 3 (* h1 h1) h2 (* h4 h4) h5) (* 6 
(* h1 h1) h2 h4 (* h5 h5)) (* 6 (* h1 h1) h2 h4 h5 h6) (* (* h1 h1) h2 (* h5 h5 
h5)) (* 4 (* h1 h1) h2 (* h5 h5) h6) (* 3 (* h1 h1) h2 h5 (* h6 h6)) (* 24 (* h1
 h1) (* h3 h3 h3) h4) (* 16 (* h1 h1) (* h3 h3 h3) h5) (* 16 (* h1 h1) (* h3 h3 
h3) h6) (* 21 (* h1 h1) (* h3 h3) (* h4 h4)) (* 46 (* h1 h1) (* h3 h3) h4 h5) 
(* 38 (* h1 h1) (* h3 h3) h4 h6) (* 16 (* h1 h1) (* h3 h3) (* h5 h5)) (* 40 (* 
h1 h1) (* h3 h3) h5 h6) (* 16 (* h1 h1) (* h3 h3) (* h6 h6)) (* 3 (* h1 h1) h3 
(* h4 h4 h4)) (* 14 (* h1 h1) h3 (* h4 h4) h5) (* 8 (* h1 h1) h3 (* h4 h4) h6) 
(* 11 (* h1 h1) h3 h4 (* h5 h5)) (* 24 (* h1 h1) h3 h4 h5 h6) (* 7 (* h1 h1) h3 
h4 (* h6 h6)) (* 2 (* h1 h1) h3 (* h5 h5 h5)) (* 12 (* h1 h1) h3 (* h5 h5) h6) 
(* 12 (* h1 h1) h3 h5 (* h6 h6)) (* 2 (* h1 h1) h3 (* h6 h6 h6)) (* (* h1 h1) 
(* h4 h4) h5 h6) (* 4 (* h1 h1) h4 (* h5 h5) h6) (* 2 (* h1 h1) h4 h5 (* h6 h6))
 (* (* h1 h1) (* h5 h5 h5) h6) (* 3 (* h1 h1) (* h5 h5) (* h6 h6)) (* (* h1 h1) 
h5 (* h6 h6 h6)) (- (* h1 (* h2 h2 h2) (* h3 h3))) (* h1 (* h2 h2 h2) h3 h5) (* 
2 h1 (* h2 h2 h2) h3 h6) (* 2 h1 (* h2 h2 h2) h4 h5) (* h1 (* h2 h2 h2) (* h5 h5
)) (* h1 (* h2 h2 h2) h5 h6) (- (* 2 h1 (* h2 h2) (* h3 h3 h3))) (* 8 h1 (* h2 
h2) (* h3 h3) h4) (* 9 h1 (* h2 h2) (* h3 h3) h5) (* 15 h1 (* h2 h2) (* h3 h3) 
h6) (* 6 h1 (* h2 h2) h3 (* h4 h4)) (* 20 h1 (* h2 h2) h3 h4 h5) (* 15 h1 (* h2 
h2) h3 h4 h6) (* 8 h1 (* h2 h2) h3 (* h5 h5)) (* 19 h1 (* h2 h2) h3 h5 h6) (* 9 
h1 (* h2 h2) h3 (* h6 h6)) (* 5 h1 (* h2 h2) (* h4 h4) h5) (* 5 h1 (* h2 h2) h4 
(* h5 h5)) (* 9 h1 (* h2 h2) h4 h5 h6) (* h1 (* h2 h2) (* h5 h5 h5)) (* 4 h1 (* 
h2 h2) (* h5 h5) h6) (* 4 h1 (* h2 h2) h5 (* h6 h6)) (* 10 h1 h2 (* h3 h3 h3) h4
) (* 8 h1 h2 (* h3 h3 h3) h5) (* 16 h1 h2 (* h3 h3 h3) h6) (* 17 h1 h2 (* h3 h3)
 (* h4 h4)) (* 49 h1 h2 (* h3 h3) h4 h5) (* 39 h1 h2 (* h3 h3) h4 h6) (* 18 h1 
h2 (* h3 h3) (* h5 h5)) (* 48 h1 h2 (* h3 h3) h5 h6) (* 22 h1 h2 (* h3 h3) (* h6
 h6)) (* 4 h1 h2 h3 (* h4 h4 h4)) (* 23 h1 h2 h3 (* h4 h4) h5) (* 13 h1 h2 h3 
(* h4 h4) h6) (* 23 h1 h2 h3 h4 (* h5 h5)) (* 46 h1 h2 h3 h4 h5 h6) (* 14 h1 h2 
h3 h4 (* h6 h6)) (* 5 h1 h2 h3 (* h5 h5 h5)) (* 21 h1 h2 h3 (* h5 h5) h6) (* 23 
h1 h2 h3 h5 (* h6 h6)) (* 5 h1 h2 h3 (* h6 h6 h6)) (* h1 h2 (* h4 h4 h4) h5) (* 
5 h1 h2 (* h4 h4) (* h5 h5)) (* 3 h1 h2 (* h4 h4) h5 h6) (* 2 h1 h2 h4 (* h5 h5 
h5)) (* 9 h1 h2 h4 (* h5 h5) h6) (* 3 h1 h2 h4 h5 (* h6 h6)) (* 2 h1 h2 (* h5 h5
 h5) h6) (* 4 h1 h2 (* h5 h5) (* h6 h6)) (* h1 h2 h5 (* h6 h6 h6)) (* 12 h1 (* 
h3 h3 h3) (* h4 h4)) (* 32 h1 (* h3 h3 h3) h4 h5) (* 20 h1 (* h3 h3 h3) h4 h6) 
(* 8 h1 (* h3 h3 h3) (* h5 h5)) (* 24 h1 (* h3 h3 h3) h5 h6) (* 8 h1 (* h3 h3 h3
) (* h6 h6)) (* 6 h1 (* h3 h3) (* h4 h4 h4)) (* 26 h1 (* h3 h3) (* h4 h4) h5) 
(* 16 h1 (* h3 h3) (* h4 h4) h6) (* 26 h1 (* h3 h3) h4 (* h5 h5)) (* 50 h1 (* h3
 h3) h4 h5 h6) (* 14 h1 (* h3 h3) h4 (* h6 h6)) (* 4 h1 (* h3 h3) (* h5 h5 h5)) 
(* 24 h1 (* h3 h3) (* h5 h5) h6) (* 24 h1 (* h3 h3) h5 (* h6 h6)) (* 4 h1 (* h3 
h3) (* h6 h6 h6)) (* 4 h1 h3 (* h4 h4 h4) h5) (* 11 h1 h3 (* h4 h4) (* h5 h5)) 
(* 12 h1 h3 (* h4 h4) h5 h6) (* 3 h1 h3 h4 (* h5 h5 h5)) (* 21 h1 h3 h4 (* h5 h5
) h6) (* 12 h1 h3 h4 h5 (* h6 h6)) (* 4 h1 h3 (* h5 h5 h5) h6) (* 10 h1 h3 (* h5
 h5) (* h6 h6)) (* 4 h1 h3 h5 (* h6 h6 h6)) (* h1 (* h4 h4) (* h5 h5) h6) (* h1 
h4 (* h5 h5 h5) h6) (* 2 h1 h4 (* h5 h5) (* h6 h6)) (* h1 (* h5 h5 h5) (* h6 h6)
) (* h1 (* h5 h5) (* h6 h6 h6)) (* (* h2 h2 h2) (* h3 h3) h4) (* (* h2 h2 h2) 
(* h3 h3) h5) (* (* h2 h2 h2) (* h3 h3) h6) (* (* h2 h2 h2) h3 (* h4 h4)) (* 2 
(* h2 h2 h2) h3 h4 h5) (* 2 (* h2 h2 h2) h3 h4 h6) (* (* h2 h2 h2) h3 (* h5 h5))
 (* 2 (* h2 h2 h2) h3 h5 h6) (* (* h2 h2 h2) h3 (* h6 h6)) (* (* h2 h2 h2) (* h4
 h4) h5) (* (* h2 h2 h2) h4 (* h5 h5)) (* 2 (* h2 h2 h2) h4 h5 h6) (* (* h2 h2 
h2) (* h5 h5) h6) (* (* h2 h2 h2) h5 (* h6 h6)) (* 2 (* h2 h2) (* h3 h3 h3) h4) 
(* 2 (* h2 h2) (* h3 h3 h3) h5) (* 2 (* h2 h2) (* h3 h3 h3) h6) (* 4 (* h2 h2) 
(* h3 h3) (* h4 h4)) (* 8 (* h2 h2) (* h3 h3) h4 h5) (* 8 (* h2 h2) (* h3 h3) h4
 h6) (* 4 (* h2 h2) (* h3 h3) (* h5 h5)) (* 8 (* h2 h2) (* h3 h3) h5 h6) (* 4 
(* h2 h2) (* h3 h3) (* h6 h6)) (* (* h2 h2) h3 (* h4 h4 h4)) (* 7 (* h2 h2) h3 
(* h4 h4) h5) (* 3 (* h2 h2) h3 (* h4 h4) h6) (* 7 (* h2 h2) h3 h4 (* h5 h5)) 
(* 14 (* h2 h2) h3 h4 h5 h6) (* 3 (* h2 h2) h3 h4 (* h6 h6)) (* (* h2 h2) h3 (* 
h5 h5 h5)) (* 7 (* h2 h2) h3 (* h5 h5) h6) (* 7 (* h2 h2) h3 h5 (* h6 h6)) (* 
(* h2 h2) h3 (* h6 h6 h6)) (* (* h2 h2) (* h4 h4 h4) h5) (* 2 (* h2 h2) (* h4 h4
) (* h5 h5)) (* 3 (* h2 h2) (* h4 h4) h5 h6) (* (* h2 h2) h4 (* h5 h5 h5)) (* 4 
(* h2 h2) h4 (* h5 h5) h6) (* 3 (* h2 h2) h4 h5 (* h6 h6)) (* (* h2 h2) (* h5 h5
 h5) h6) (* 2 (* h2 h2) (* h5 h5) (* h6 h6)) (* (* h2 h2) h5 (* h6 h6 h6)) (* 4 
h2 (* h3 h3 h3) (* h4 h4)) (* 8 h2 (* h3 h3 h3) h4 h5) (* 8 h2 (* h3 h3 h3) h4 
h6) (* 4 h2 (* h3 h3 h3) (* h5 h5)) (* 8 h2 (* h3 h3 h3) h5 h6) (* 4 h2 (* h3 h3
 h3) (* h6 h6)) (* 2 h2 (* h3 h3) (* h4 h4 h4)) (* 14 h2 (* h3 h3) (* h4 h4) h5)
 (* 6 h2 (* h3 h3) (* h4 h4) h6) (* 14 h2 (* h3 h3) h4 (* h5 h5)) (* 28 h2 (* h3
 h3) h4 h5 h6) (* 6 h2 (* h3 h3) h4 (* h6 h6)) (* 2 h2 (* h3 h3) (* h5 h5 h5)) 
(* 14 h2 (* h3 h3) (* h5 h5) h6) (* 14 h2 (* h3 h3) h5 (* h6 h6)) (* 2 h2 (* h3 
h3) (* h6 h6 h6)) (* 4 h2 h3 (* h4 h4 h4) h5) (* 8 h2 h3 (* h4 h4) (* h5 h5)) 
(* 12 h2 h3 (* h4 h4) h5 h6) (* 4 h2 h3 h4 (* h5 h5 h5)) (* 16 h2 h3 h4 (* h5 h5
) h6) (* 12 h2 h3 h4 h5 (* h6 h6)) (* 4 h2 h3 (* h5 h5 h5) h6) (* 8 h2 h3 (* h5 
h5) (* h6 h6)) (* 4 h2 h3 h5 (* h6 h6 h6)) (* h2 (* h4 h4 h4) (* h5 h5)) (* h2 
(* h4 h4) (* h5 h5 h5)) (* 3 h2 (* h4 h4) (* h5 h5) h6) (* 2 h2 h4 (* h5 h5 h5) 
h6) (* 3 h2 h4 (* h5 h5) (* h6 h6)) (* h2 (* h5 h5 h5) (* h6 h6)) (* h2 (* h5 h5
) (* h6 h6 h6)) (* 8 (* h3 h3 h3) (* h4 h4) h5) (* 8 (* h3 h3 h3) h4 (* h5 h5)) 
(* 16 (* h3 h3 h3) h4 h5 h6) (* 8 (* h3 h3 h3) (* h5 h5) h6) (* 8 (* h3 h3 h3) 
h5 (* h6 h6)) (* 4 (* h3 h3) (* h4 h4 h4) h5) (* 8 (* h3 h3) (* h4 h4) (* h5 h5)
) (* 12 (* h3 h3) (* h4 h4) h5 h6) (* 4 (* h3 h3) h4 (* h5 h5 h5)) (* 16 (* h3 
h3) h4 (* h5 h5) h6) (* 12 (* h3 h3) h4 h5 (* h6 h6)) (* 4 (* h3 h3) (* h5 h5 h5
) h6) (* 8 (* h3 h3) (* h5 h5) (* h6 h6)) (* 4 (* h3 h3) h5 (* h6 h6 h6)) (* 2 
h3 (* h4 h4 h4) (* h5 h5)) (* 2 h3 (* h4 h4) (* h5 h5 h5)) (* 6 h3 (* h4 h4) (* 
h5 h5) h6) (* 4 h3 h4 (* h5 h5 h5) h6) (* 6 h3 h4 (* h5 h5) (* h6 h6)) (* 2 h3 
(* h5 h5 h5) (* h6 h6)) (* 2 h3 (* h5 h5) (* h6 h6 h6))) 0)))
(check-sat)
(exit)
