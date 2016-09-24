//
//  PardisoSolver.h
//
//  Created by Olga Diamanti on 07/01/15.
//  Copyright (c) 2015 Olga Diamanti. All rights reserved.
//

#ifndef _PardisoSolver__
#define _PardisoSolver__

#include <vector>
#include <Eigen/Core>


 extern "C" {
 /* PARDISO prototype. */
 void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
 void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                   double *, int    *,    int *, int *,   int *, int *,
                   int *, double *, double *, int *, double *);
 void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
 void pardiso_chkvec     (int *, int *, double *, int *);
 void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                          double *, int *);
 }

template <typename vectorTypeI, typename vectorTypeS>
 class PardisoSolver
 {
 public:
   
   PardisoSolver() ;
   ~PardisoSolver();
   
   void set_type(int _mtype);
   
   void init();

   void set_pattern(const vectorTypeI &II,
                    const vectorTypeI &JJ,
                    const vectorTypeS SS);
   void analyze_pattern();
   
   bool factorize();
   
   void solve(Eigen::VectorXd &rhs,
              Eigen::VectorXd &result);

   void update_a(const vectorTypeS &SS);

 protected:
   Eigen::VectorXi ia, ja;
   std::vector<Eigen::VectorXi> iis;
   Eigen::VectorXd a;
   int numRows;

   //pardiso stuff
   /*
    1: real and structurally symmetric, supernode pivoting
    2: real and symmetric positive definite
    -2: real and symmetric indefinite, diagonal or Bunch-Kaufman pivoting
    11: real and nonsymmetric, complete supernode pivoting
    */
   int mtype;       /* Matrix Type */

   // Remember if matrix is symmetric or not, to
   // decide whether to eliminate the non-upper-
   // diagonal entries from the input II,JJ,SS
   bool is_symmetric;
   
   int nrhs = 1;     /* Number of right hand sides. */
   /* Internal solver memory pointer pt, */
   /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
   /* or void *pt[64] should be OK on both architectures */
   void *pt[64];
   /* Pardiso control parameters. */
   int iparm[64];
   double   dparm[64];
   int maxfct, mnum, phase, error, msglvl, solver =0;
   /* Number of processors. */
   int      num_procs;
   /* Auxiliary variables. */
   char    *var;
   int i, k;
   double ddum;          /* Double dummy */
   int idum;         /* Integer dummy. */

};


#endif /* defined(_PardisoSolver__) */