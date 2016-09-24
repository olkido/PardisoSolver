//
//  PardisoSolver.cpp
//
//  Created by Olga Diamanti on 07/01/15.
//  Copyright (c) 2015 Olga Diamanti. All rights reserved.
//

#include "PardisoSolver.h"
#include <igl/sortrows.h>
#include <igl/unique.h>
#include <igl/matlab_format.h>


using namespace std;
//#define PLOTS_PARDISO


template <typename vectorTypeI, typename vectorTypeS>
PardisoSolver<vectorTypeI,vectorTypeS>::PardisoSolver():
mtype(-1)
{}

template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::set_type(int _mtype)
{
  if ((_mtype !=-2) && (_mtype !=2) && (_mtype !=1) && (_mtype !=11))
  {
    printf("Pardiso mtype %d not supported.",_mtype);
    exit(1);
  }
  mtype = _mtype;
  // As per https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/283738
  // structurally symmetric need the full matrix
  // structure to be passed, not only the upper
  // diagonal part. So is_symmetric should be set to
  // false in that case.
  is_symmetric = (mtype ==2) ||(mtype ==-2);
  init();
}

template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::init()
{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  /* -------------------------------------------------------------------- */
  /* ..  Setup Pardiso control parameters.                                */
  /* -------------------------------------------------------------------- */
  
  error = 0;
  solver=0;/* use sparse direct solver */
  pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);
  
  if (error != 0)
  {
    if (error == -10 )
      printf("No license file found \n");
    if (error == -11 )
      printf("License is expired \n");
    if (error == -12 )
      printf("Wrong username or hostname \n");
    exit(1);
  }
  else
    printf("[PARDISO]: License check was successful ... \n");
  
  
  /* Numbers of processors, value of OMP_NUM_THREADS */
  var = getenv("OMP_NUM_THREADS");
  if(var != NULL)
    sscanf( var, "%d", &num_procs );
  else {
    printf("Set environment OMP_NUM_THREADS to 1");
    exit(1);
  }
  iparm[2]  = num_procs;
  
  maxfct = 1;		/* Maximum number of numerical factorizations.  */
  mnum   = 1;         /* Which factorization to use. */
  
  msglvl = 0;         /* Print statistical information  */
  error  = 0;         /* Initialize error flag */
  
  
  //  /* -------------------------------------------------------------------- */
  //  /* .. Initialize the internal solver memory pointer. This is only */
  //  /* necessary for the FIRST call of the PARDISO solver. */
  //  /* -------------------------------------------------------------------- */
  //  for ( i = 0; i < 64; i++ )
  //  {
  //    pt[i] = 0;
  //  }
  
  
}

template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::update_a(const vectorTypeS &SS)
{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  for (int i=0; i<a.rows(); ++i)
  {
    a(i) = 0;
    for (int j=0; j<iis[i].size(); ++j)
      a(i) += SS[iis[i](j)];
  }
}

//todo: make sure diagonal terms are included, even as zeros (pardiso claims this is necessary for best performance)
template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::set_pattern(const vectorTypeI &II,
                                                         const vectorTypeI &JJ,
                                                         const vectorTypeS SS)


{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  numRows = 0;
  for (int i=0; i<II.size(); ++i)
    if (II[i] > numRows )
      numRows = II[i];
  numRows ++;
  
  vectorTypeS SS_true = SS;
  Eigen::MatrixXi M0;
  //if the matrix is symmetric, only store upper triangular part
  if (is_symmetric)
  {
    std::vector<int> pick;
    pick.reserve(II.size()/2);
    for (int i = 0; i<II.size();++i)
      if (II[i]<=JJ[i])
        pick.push_back(i);
    M0.resize(pick.size(),3);
    SS_true.resize(pick.size(),1);
    for (int i = 0; i<pick.size();++i)
    {
      M0.row(i)<< II[pick[i]], JJ[pick[i]], i;
      SS_true[i] = SS[pick[i]];
    }
  }
  else
  {
    M0.resize(II.size(),3);
    for (int i = 0; i<II.size();++i)
      M0.row(i)<< II[i], JJ[i], i;
  }
  
  //temps
  Eigen::MatrixXi t;
  Eigen::VectorXi tI;
  
  Eigen::MatrixXi M_;
  igl::sortrows(M0, true, M_, tI);
  
  int si,ei,currI;
  si = 0;
  while (si<M_.rows())
  {
    currI = M_(si,0);
    ei = si;
    while (ei<M_.rows() && M_(ei,0) == currI)
      ++ei;
    igl::sortrows(M_.block(si, 1, ei-si, 2).eval(), true, t, tI);
    M_.block(si, 1, ei-si, 2) = t;
    si = ei;
  }
  
  Eigen::MatrixXi M;
  Eigen::VectorXi IM_;
  igl::unique_rows(M_.leftCols(2).eval(), M, IM_, tI);
  int numUniqueElements = M.rows();
  iis.resize(numUniqueElements);
  for (int i=0; i<numUniqueElements; ++i)
  {
    si = IM_(i);
    if (i<numUniqueElements-1)
      ei = IM_(i+1);
    else
      ei = M_.rows();
    iis[i] = M_.block(si, 2, ei-si, 1);
  }
  
  a.resize(numUniqueElements, 1);
  for (int i=0; i<numUniqueElements; ++i)
  {
    a(i) = 0;
    for (int j=0; j<iis[i].size(); ++j)
      a(i) += SS_true[iis[i](j)];
  }
  
  // now M_ and elements in sum have the row, column and indices in sum of the
  // unique non-zero elements in B1
  ia.setZero(numRows+1,1);ia(numRows) = numUniqueElements+1;
  ja = M.col(1).array()+1;
  currI = -1;
  for (int i=0; i<numUniqueElements; ++i)
  {
    if(currI != M(i,0))
    {
      ia(M(i,0)) = i+1;//do not subtract 1
      currI = M(i,0);
    }
  }
  
#ifdef PLOTS_PARDISO
  printf("ia: ");
  for (int i=0; i<ia.size(); ++i)
    printf("%d ",ia[i]);
  printf("\n\n");
  
  printf("ja: ");
  for (int i=0; i<ja.size(); ++i)
    printf("%d ",ja[i]);
  printf("\n\n");
  
  printf("a: ");
  for (int i=0; i<a.size(); ++i)
    printf("%.2f ",a[i]);
  printf("\n\n");
#endif
  
  // matrix in CRS can be expressed with ia, ja and iis
  
}

template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::analyze_pattern()
{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  
#ifdef PLOTS_PARDISO
  /* -------------------------------------------------------------------- */
  /*  .. pardiso_chk_matrix(...)                                          */
  /*     Checks the consistency of the given matrix.                      */
  /*     Use this functionality only for debugging purposes               */
  /* -------------------------------------------------------------------- */
  
  pardiso_chkmatrix  (&mtype, &numRows, a.data(), ia.data(), ja.data(), &error);
  if (error != 0) {
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
  }
#endif
  /* -------------------------------------------------------------------- */
  /* ..  Reordering and Symbolic Factorization.  This step also allocates */
  /*     all memory that is necessary for the factorization.              */
  /* -------------------------------------------------------------------- */
  phase = 11;
  
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error, dparm);
  
  if (error != 0) {
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);
  }
#ifdef PLOTS_PARDISO
  printf("\nReordering completed ... ");
  printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
  printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
#endif
  
}

template <typename vectorTypeI, typename vectorTypeS>
bool PardisoSolver<vectorTypeI,vectorTypeS>::factorize()
{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  /* -------------------------------------------------------------------- */
  /* ..  Numerical factorization.                                         */
  /* -------------------------------------------------------------------- */
  phase = 22;
  //  iparm[32] = 1; /* compute determinant */
  
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error,  dparm);
  
  if (error != 0) {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }
#ifdef PLOTS_PARDISO
  printf ("\nFactorization completed ... ");
#endif
  return (error ==0);
}

template <typename vectorTypeI, typename vectorTypeS>
void PardisoSolver<vectorTypeI,vectorTypeS>::solve(Eigen::VectorXd &rhs,
                                                   Eigen::VectorXd &result)
{
  if (mtype ==-1)
  {
    printf("Pardiso mtype not set.");
    exit(1);
  }
  
#ifdef PLOTS_PARDISO
  /* -------------------------------------------------------------------- */
  /* ..  pardiso_chkvec(...)                                              */
  /*     Checks the given vectors for infinite and NaN values             */
  /*     Input parameters (see PARDISO user manual for a description):    */
  /*     Use this functionality only for debugging purposes               */
  /* -------------------------------------------------------------------- */
  
  pardiso_chkvec (&numRows, &nrhs, rhs.data(), &error);
  if (error != 0) {
    printf("\nERROR  in right hand side: %d", error);
    exit(1);
  }
  
  /* -------------------------------------------------------------------- */
  /* .. pardiso_printstats(...)                                           */
  /*    prints information on the matrix to STDOUT.                       */
  /*    Use this functionality only for debugging purposes                */
  /* -------------------------------------------------------------------- */
  
  pardiso_printstats (&mtype, &numRows, a.data(), ia.data(), ja.data(), &nrhs, rhs.data(), &error);
  if (error != 0) {
    printf("\nERROR right hand side: %d", error);
    exit(1);
  }
  
#endif
  result.resize(numRows, 1);
  /* -------------------------------------------------------------------- */
  /* ..  Back substitution and iterative refinement.                      */
  /* -------------------------------------------------------------------- */
  phase = 33;
  
  iparm[7] = 1;       /* Max numbers of iterative refinement steps. */
  
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
           iparm, &msglvl, rhs.data(), result.data(), &error,  dparm);
  
  if (error != 0) {
    printf("\nERROR during solution: %d", error);
    exit(3);
  }
#ifdef PLOTS_PARDISO
  printf("\nSolve completed ... ");
  printf("\nThe solution of the system is: ");
  for (i = 0; i < numRows; i++) {
    printf("\n x [%d] = % f", i, result.data()[i] );
  }
  printf ("\n\n");
#endif
}

template <typename vectorTypeI, typename vectorTypeS>
PardisoSolver<vectorTypeI,vectorTypeS>::~PardisoSolver()
{
  if (mtype == -1)
    return;
  /* -------------------------------------------------------------------- */
  /* ..  Termination and release of memory.                               */
  /* -------------------------------------------------------------------- */
  phase = -1;                 /* Release internal memory. */
  
  pardiso (pt, &maxfct, &mnum, &mtype, &phase,
           &numRows, &ddum, ia.data(), ja.data(), &idum, &nrhs,
           iparm, &msglvl, &ddum, &ddum, &error,  dparm);
}

template class PardisoSolver<std::vector<int, std::allocator<int> >, std::vector<double, std::allocator<double> > >;

template class PardisoSolver<Eigen::Matrix<int, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >;

