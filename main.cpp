//
//  main.cpp
//  
//
//  Created by Olga Diamanti on 23/09/16.
//
//

#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <unsupported/eigen/MatrixFunctions>

#include "PardisoSolver.h"
#include <igl/floor.h>
#include <igl/list_to_matrix.h>
#include <igl/sparse.h>

using namespace std;
using namespace Eigen;

//both of these two options should work
#if 1
  typedef vector<int> vectorTypeI;
  typedef vector<double> vectorTypeS;
#else
  typedef VectorXi vectorTypeI;
  typedef VectorXd vectorTypeS;
#endif

double nnz_pc = 0.1;

//needed for all tests: a sparse, square, non-symmetric, full rank matrix
void create_square_fullrank_matrix(const int N, MatrixXd &mat)
{
 
  //number of non-zeros (approximately)
  int nnz = nnz_pc*N*N;

  SparseMatrix<double> T;
  VectorXi II,JJ;
  VectorXd SS;
  

  // first create a generic sparse matrix, by
  // creating nnz random indices from 0 to N-1, nnz random values
  igl::floor((0.5*(VectorXd::Random(nnz,1).array()+1.)*N).eval(),II);
  igl::floor((0.5*(VectorXd::Random(nnz,1).array()+1.)*N).eval(),JJ);
  SS = 0.5*(VectorXd::Random(nnz,1).array()+1.);
  igl::sparse(II, JJ, SS, N, N, T);
  
  mat = MatrixXd(T);

  // mat is not guaranteed to be full rank. For that, use the matrix exponential trick found in
  // http://math.stackexchange.com/questions/273061/how-to-randomly-construct-a-square-full-ranked-matrix-with-low-determinant
  MatrixExponential<MatrixXd> expo(mat);
  expo.compute(mat);
  
}

//extract II,JJ,SS (row,column and value vectors) from sparse matrix, std::vector version
void extract_ij_from_matrix(const Eigen::SparseMatrix<double> &A,
                            vector<int> &II,
                            vector<int> &JJ,
                            vector<double> &SS)
{
  for (int k=0; k<A.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    {
      double ss = it.value();
      int ii = it.row();   // row index
      int jj = it.col();   // row index
      {
        II.push_back(ii);
        JJ.push_back(jj);
        SS.push_back(ss);
      }
    }
}

//extract II,JJ,SS (row,column and value vectors) from sparse matrix, Eigen version
void extract_ij_from_matrix(const Eigen::SparseMatrix<double> &A,
                            VectorXi &II,
                            VectorXi &JJ,
                            VectorXd &SS)
{
  II.resize(A.nonZeros());
  JJ.resize(A.nonZeros());
  SS.resize(A.nonZeros());
  int ind = 0;
  for (int k=0; k<A.outerSize(); ++k)
    for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
    {
      double ss = it.value();
      int ii = it.row();   // row index
      int jj = it.col();   // row index
      {
        II[ind] = ii;
        JJ[ind] = jj;
        SS[ind] = ss;
        ind ++;
      }
    }
}

//shuffle entries in II,JJ,SS (to show invariance to order of columns/rows)
void shuffle(vectorTypeI &II,
             vectorTypeI &JJ,
             vectorTypeS &SS)
{
  vectorTypeI II_ori = II;
  vectorTypeI JJ_ori = JJ;
  vectorTypeS SS_ori = SS;
  
  std::vector<int> perm;
  for (int i=0; i<II.size(); ++i) {
    perm.push_back(i);
  }
  std::random_shuffle(perm.begin(), perm.end());

  for (int i=0; i<perm.size(); ++i)
  {
    II[i] = II_ori[perm[i]];
    JJ[i] = JJ_ori[perm[i]];
    SS[i] = SS_ori[perm[i]];
  }

}

//from input II,JJ,SS create a new set II,JJ,SS of twice the size
//by appending a shuffled copy of the original II,JJ,SS at the bottom
//(to show invariance to repeated (column,row) entries)
void shuffle_and_append(vectorTypeI &II,
                        vectorTypeI &JJ,
                        vectorTypeS &SS)
{
  int num = II.size();
  vectorTypeI II_ori = II;
  vectorTypeI JJ_ori = JJ;
  vectorTypeS SS_ori = SS;

  //shuffle to get a second copy
  vectorTypeI IIa = II;
  vectorTypeI JJa = JJ;
  vectorTypeS SSa = SS;
  shuffle(IIa, JJa, SSa);
  
  //append
  II.resize(2*num);
  JJ.resize(2*num);
  SS.resize(2*num);
  
  std::vector<int> perm;
  for (int i=0; i<num; ++i)
  {
    II[i] = II_ori[i];
    JJ[i] = JJ_ori[i];
    SS[i] = SS_ori[i];
  }
  for (int i=0; i<num; ++i)
  {
    II[num+i] = IIa[i];
    JJ[num+i] = JJa[i];
    SS[num+i] = SSa[i];
  }
  
}

//make sparse matrix from II,JJ,SS - Eigen version
void make_sparse_matrix(const VectorXi &II,
                        const VectorXi &JJ,
                        const VectorXd &SS,
                        Eigen::SparseMatrix<double> &A)
{
  igl::sparse(II,JJ,SS,A);
}


//make sparse matrix from II,JJ,SS - std::vector version
void make_sparse_matrix(const vector<int> &II_vec,
                        const vector<int> &JJ_vec,
                        const vector<double> &SS_vec,
                        Eigen::SparseMatrix<double> &A)
{
  VectorXi II;
  VectorXi JJ;
  VectorXd SS;
  igl::list_to_matrix(II_vec, II);
  igl::list_to_matrix(JJ_vec, JJ);
  igl::list_to_matrix(SS_vec, SS);
  igl::sparse(II,JJ,SS,A);
}

//general test: uses igl::sparse, and permutations and shuffles to create a square, non-symmetric full rank matrix
//II,JJ can be in any order and will contain duplicate entries (in which case the corresponding elements of SS will be summed into the matrix)
void run_general_test(const int N)
{
  //matrix, Eigen-style
  SparseMatrix<double> A;
  
  //matrix, pardiso-style
  vectorTypeI II, JJ;
  vectorTypeS SS;
  
  //solvers, Eigen-style
  SparseLU<SparseMatrix<double> > lu;
  bool success_eigen;
  double error_eigen;
  
  //solver, pardiso
  PardisoSolver<vectorTypeI, vectorTypeS> pardiso;
  bool success_pardiso;
  double error_pardiso;
  
  //rhs vector
  VectorXd b;
  
  //solution vector
  VectorXd x_eigen, x_pardiso;
  
  //ground truth solution
  VectorXd gt;
  
  //create matrix
  MatrixXd mat;
  create_square_fullrank_matrix(N, mat);
  A = mat.eval().sparseView();
  A.makeCompressed();

  extract_ij_from_matrix(A, II, JJ, SS);
  shuffle_and_append(II,JJ,SS);
  make_sparse_matrix(II,JJ,SS,A);
  
  
  // make pardiso matrix
  pardiso.set_type(11);
  pardiso.set_pattern(II, JJ, SS);
  
  
  //create ground truth solution
  gt.setRandom(N,1);
  
  //set rhs
  b = A*gt;
  
  //solve - Eigen
  lu.compute(A);
  success_eigen = lu.info() == Success;
  cerr<<"Eigen factorization was successful: "<<success_eigen<<endl;
  x_eigen = lu.solve(b);
  error_eigen = (A*x_eigen - b).cwiseAbs().maxCoeff();
  cerr<<"Eigen error: "<<error_eigen<<endl;
  
  
  //solve - pardiso
  pardiso.analyze_pattern();
  success_pardiso = pardiso.factorize();
  cerr<<"Pardiso factorization was successful: "<<success_pardiso<<endl;
  pardiso.solve(b, x_pardiso);
  error_pardiso = (A*x_pardiso - b).cwiseAbs().maxCoeff();
  cerr<<"Pardiso error: "<<error_pardiso<<endl;
  
}

//square, non symmetric test, without duplicate entries in II,JJ
void run_ns_test(const int N)
{
  //matrix, Eigen-style
  SparseMatrix<double> A;
  
  //matrix, pardiso-style
  vectorTypeI II, JJ;
  vectorTypeS SS;
  
  //solvers, Eigen-style
  SparseLU<SparseMatrix<double> > lu;
  bool success_eigen;
  double error_eigen;
  
  //solver, pardiso
  PardisoSolver<vectorTypeI, vectorTypeS> pardiso;
  bool success_pardiso;
  double error_pardiso;
  
  //rhs vector
  VectorXd b;
  
  //solution vector
  VectorXd x_eigen, x_pardiso;
  
  //ground truth solution
  VectorXd gt;
  
  //create matrix
  MatrixXd mat;
  create_square_fullrank_matrix(N, mat);
  A = mat.eval().sparseView();
  A.makeCompressed();
  
  
  // make pardiso matrix
  pardiso.set_type(11);
  extract_ij_from_matrix(A, II, JJ, SS);
  pardiso.set_pattern(II, JJ, SS);
  
  
  //create ground truth solution
  gt.setRandom(N,1);

  //set rhs
  b = A*gt;
  
  //solve - Eigen
  lu.compute(A);
  success_eigen = lu.info() == Success;
  cerr<<"Eigen factorization was successful: "<<success_eigen<<endl;
  x_eigen = lu.solve(b);
  error_eigen = (A*x_eigen - b).cwiseAbs().maxCoeff();
  cerr<<"Eigen error: "<<error_eigen<<endl;
  
  
  //solve - pardiso
  pardiso.analyze_pattern();
  success_pardiso = pardiso.factorize();
  cerr<<"Pardiso factorization was successful: "<<success_pardiso<<endl;
  pardiso.solve(b, x_pardiso);
  error_pardiso = (A*x_pardiso - b).cwiseAbs().maxCoeff();
  cerr<<"Pardiso error: "<<error_pardiso<<endl;
  
}

//square, symmetric indefinite (in fact, negative) test, without duplicate entries in II,JJ
void run_si_test(const int N)
{
  //matrix, Eigen-style
  SparseMatrix<double> A;
  
  //matrix, pardiso-style
  vectorTypeI II, JJ;
  vectorTypeS SS;
  
  //solvers, Eigen-style
  SparseLU<SparseMatrix<double> > lu;
  bool success_eigen;
  double error_eigen;
  
  //solver, pardiso
  PardisoSolver<vectorTypeI, vectorTypeS> pardiso;
  bool success_pardiso;
  double error_pardiso;
  
  //rhs vector
  VectorXd b;
  
  //solution vector
  VectorXd x_eigen, x_pardiso;
  
  //ground truth solution
  VectorXd gt;
  
  //create matrix
  MatrixXd mat;
  create_square_fullrank_matrix(N, mat);
  A = (-1.*mat.transpose()*mat).eval().sparseView();
  A.makeCompressed();
  
  
  // make pardiso matrix
  pardiso.set_type(-2);
  extract_ij_from_matrix(A, II, JJ, SS);
  pardiso.set_pattern(II, JJ, SS);
  
  //create ground truth solution
  gt.setRandom(N,1);
  
  //set rhs
  b = A*gt;
  
  //solve - Eigen
  lu.compute(A);
  success_eigen = lu.info() == Success;
  cerr<<"Eigen factorization was successful: "<<success_eigen<<endl;
  x_eigen = lu.solve(b);
  error_eigen = (A*x_eigen - b).cwiseAbs().maxCoeff();
  cerr<<"Eigen error: "<<error_eigen<<endl;
  
  
  //solve - pardiso
  pardiso.analyze_pattern();
  success_pardiso = pardiso.factorize();
  cerr<<"Pardiso factorization was successful: "<<success_pardiso<<endl;
  pardiso.solve(b, x_pardiso);
  error_pardiso = (A*x_pardiso - b).cwiseAbs().maxCoeff();
  cerr<<"Pardiso error: "<<error_pardiso<<endl;
  
}

//square, symmetric positive definite test, without duplicate entries in II,JJ
void run_spd_test(const int N)
{
  //matrix, Eigen-style
  SparseMatrix<double> A;
  
  //matrix, pardiso-style
  vectorTypeI II, JJ;
  vectorTypeS SS;

  //solvers, Eigen-style
  SimplicialLDLT<SparseMatrix<double> > ldlt;
  bool success_eigen;
  double error_eigen;

  //solver, pardiso
  PardisoSolver<vectorTypeI, vectorTypeS> pardiso;
  bool success_pardiso;
  double error_pardiso;

  //rhs vector
  VectorXd b;

  //solution vector
  VectorXd x_eigen, x_pardiso;

  //ground truth solution
  VectorXd gt;
  
  //create matrix
  MatrixXd mat;
  create_square_fullrank_matrix(N, mat);
  A = (mat.transpose()*mat).eval().sparseView();
  A.makeCompressed();
  
  
  // make pardiso matrix
  pardiso.set_type(2);
  extract_ij_from_matrix(A, II, JJ, SS);
  pardiso.set_pattern(II, JJ, SS);
  
  
  //create ground truth solution
  gt.setRandom(N,1);

  //set rhs
  b = A*gt;
  
  //solve - Eigen
  ldlt.compute(A);
  success_eigen = ldlt.info() == Success;
  cerr<<"Eigen factorization was successful: "<<success_eigen<<endl;
  x_eigen = ldlt.solve(b);
  error_eigen = (A*x_eigen - b).cwiseAbs().maxCoeff();
  cerr<<"Eigen error: "<<error_eigen<<endl;
  
  
  //solve - pardiso
  pardiso.analyze_pattern();
  success_pardiso = pardiso.factorize();
  cerr<<"Pardiso factorization was successful: "<<success_pardiso<<endl;
  pardiso.solve(b, x_pardiso);
  error_pardiso = (A*x_pardiso - b).cwiseAbs().maxCoeff();
  cerr<<"Pardiso error: "<<error_pardiso<<endl;

}

int main(int argc, char *argv[])
{
  srand (time(NULL));
  

  //sparse matrix size
  int N = 10;
  
  
  cerr<<endl<<endl<< "***** Test 1: full-rank, square, symmetric, positive definite *****"<<endl;
  run_spd_test(N);

  cerr<<endl<<endl<< "***** Test 2: full-rank, square, symmetric, indefinite (negative definite) *****"<<endl;
  run_si_test(N);
  
  cerr<<endl<<endl<< "***** Test 3: full-rank, square, nonsymmetric *****"<<endl;
  run_ns_test(N);
  
  cerr<<endl<<endl<< "***** Test 4: full-rank, square, nonsymmetric, with igl::sparse and random repetitions of indices (most general case) *****"<<endl;
  run_general_test(N);

  return 0;

}
