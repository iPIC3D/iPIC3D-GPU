/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GMRES.h"
#include "Alloc.h"
#include "Basic.h"
#include "TimeTasks.h"
#include "errors.h"
#include "parallel.h"
#include <mpi.h>
// #include "ipicdefs.h"
#include "EMfields3D.h"
#include "VCtopology3D.h"

#include <cstring>

// Flexible-GMRES --> allows to a have a flexible preeconditioner
void FGMRES(FIELD_IMAGE FunctionImage, FIELD_IMAGE LocalFunctionImage,
            double* xkrylov, int xkrylovlen, const double* b, int m,
            int max_iter, double tol, EMfields3D* field) {
  if (m > xkrylovlen) {
    // m need not be the same for all processes,
    // we cannot restrict this test to the main process,
    // (although we could probably restrict it to the
    // process with the highest cartesian rank).
    eprintf("In GMRES the dimension of Krylov space(m) "
            "can't be > (length of krylov vector)/(# processors)\n");
  }
  // std::cout<< "FGMRES init" <<std::endl;
  bool GMRESVERBOSE = false;
  double initial_error, rho_tol;
  double* r = new double[xkrylovlen];
  double* im = new double[xkrylovlen];

  double* s = new double[m + 1];
  double* cs = new double[m + 1];
  double* sn = new double[m + 1];
  double* y = new double[m + 3];
  eqValue(0.0, s, m + 1);
  eqValue(0.0, cs, m + 1);
  eqValue(0.0, sn, m + 1);
  eqValue(0.0, y, m + 3);

  // preconditioner parameters and arrays
  const double tolPrec = tol > 0.001 ? 0.1 : tol * 10;
  const int mPrec = 30;
  double* rPrec = new double[xkrylovlen];
  double* imPrec = new double[xkrylovlen];

  double* sPrec = new double[mPrec + 1];
  double* csPrec = new double[mPrec + 1];
  double* snPrec = new double[mPrec + 1];
  double* yPrec = new double[mPrec + 3];

  double** HPrec = newArr2(double, mPrec + 1, mPrec);
  double** VPrec = newArr2(double, mPrec + 1, xkrylovlen);

  // allocate H for storing the results from decomposition
  double** H = newArr2(double, m + 1, m);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < m; jj++)
      H[ii][jj] = 0;
  // allocate V
  double** V = newArr2(double, m + 1, xkrylovlen);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < xkrylovlen; jj++)
      V[ii][jj] = 0;

  // allocate Z
  double** Z = newArr2(double, m, xkrylovlen);
  for (int ii = 0; ii < m; ii++)
    for (int jj = 0; jj < xkrylovlen; jj++)
      Z[ii][jj] = 0;

  if (GMRESVERBOSE && is_output_thread()) {
    printf("------------------------------------\n"
           "-             FGMRES                -\n"
           "------------------------------------\n\n");
  }

  MPI_Comm fieldcomm = (field->get_vct()).getFieldComm();

  double normb = normP(b, xkrylovlen, &fieldcomm);
  if (normb == 0.0)
    normb = 1.0;
  int itr = 0;
  for (itr = 0; itr < max_iter; itr++) {
    // std::cout<< "FGMRES for loop iter: "<< itr <<std::endl;
    //  r = b - A*x
    (field->*FunctionImage)(im, xkrylov);
    sub(r, b, im, xkrylovlen);
    initial_error = normP(r, xkrylovlen, &fieldcomm);

    if (itr == 0) {
      if (is_output_thread())
        printf("Initial residual: %g norm b vector (source) = %g\n",
               initial_error, normb);
      // cout << "Initial residual: " << initial_error << " norm b vector
      // (source) = " << normb << endl;
      rho_tol = initial_error * tol;

      if ((initial_error / normb) <= tol) {
        if (is_output_thread())
          printf("GMRES converged without iterations: initial error < "
                 "tolerance\n");
        // cout << "GMRES converged without iterations: initial error <
        // tolerance" << endl;
        break;
      }
    }

    scale(V[0], r, (1.0 / initial_error), xkrylovlen);
    eqValue(0.0, s, m + 1);
    s[0] = initial_error;
    int k = 0;
    // std::cout<< "FGMRES for loop iter: "<< itr <<std::endl;
    while (rho_tol < initial_error && k < m) {

      // Z(:,k) = A^-1 V(:,k)
      // GMRESasPreconditionerNoComm(&Field::MaxwellImageLocal, Z[k],
      // xkrylovlen, V[k], 50, 200, tol/1000, field);
      GMRESasPreconditioner(&Field::MaxwellImage, Z[k], xkrylovlen, V[k], mPrec,
                            max_iter, tolPrec, field, rPrec, imPrec, sPrec,
                            csPrec, snPrec, yPrec, HPrec, VPrec);
      // memcpy(Z[k],V[k], sizeof(double)*xkrylovlen);
      //  w= A*Z(:,k)
      double* w = V[k + 1];
      (field->*FunctionImage)(w, Z[k]);

      for (int j = 0; j <= k; j++) {
        y[j] = dot(w, V[j], xkrylovlen);
      }
      y[k + 1] = norm2(w, xkrylovlen);
      {

        MPI_Allreduce(MPI_IN_PLACE, y, (k + 2), MPI_DOUBLE, MPI_SUM, fieldcomm);
      }

      for (int j = 0; j <= k; j++) {
        H[j][k] = y[j];
        addscale(-H[j][k], V[k + 1], V[j], xkrylovlen);
      }
      // Is there a numerically stable way to
      // eliminate this second all-reduce all?
      H[k + 1][k] = normP(V[k + 1], xkrylovlen, &fieldcomm);
      // std::cout<< "FGMRES  H[k+1][k]: "<< H[k+1][k] << "k: "<< k <<std::endl;
      //
      //  check that vectors are orthogonal
      //
      // for (register int j = 0; j <= k; j++) {
      //   dprint(dotP(w, V[j], xkrylovlen));
      // }

      double av = sqrt(y[k + 1]);
      // why are we testing floating point numbers
      // for equality?  Is this supposed to say
      // if (av < delta * fabs(H[k + 1][k]))
      const double delta = 0.001;
      if (av + delta * H[k + 1][k] == av) {
        for (int j = 0; j <= k; j++) {
          const double htmp = dotP(w, V[j], xkrylovlen, &fieldcomm);
          H[j][k] = H[j][k] + htmp;
          addscale(-htmp, w, V[j], xkrylovlen);
        }
        H[k + 1][k] = normP(w, xkrylovlen, &fieldcomm);
      }
      // normalize the new vector
      scale(w, (1.0 / H[k + 1][k]), xkrylovlen);

      if (0 < k) {

        for (int j = 0; j < k; j++)
          ApplyPlaneRotation(H[j + 1][k], H[j][k], cs[j], sn[j]);

        getColumn(y, H, k, m + 1);
      }

      const double mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k] = H[k][k] / mu;
      sn[k] = -H[k + 1][k] / mu;
      H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      ApplyPlaneRotation(s[k + 1], s[k], cs[k], sn[k]);
      initial_error = fabs(s[k]);
      k++;
    }
    k--;
    y[k] = s[k] / H[k][k];

    for (int i = k - 1; i >= 0; i--) {
      double tmp = 0.0;
      for (int l = i + 1; l <= k; l++)
        tmp += H[i][l] * y[l];
      y[i] = (s[i] - tmp) / H[i][i];
    }

    for (int j = 0; j < k; j++) {
      const double yj = y[j];
      // double* Vj = V[j];
      double* Zj = Z[j];
      for (int i = 0; i < xkrylovlen; i++)
        xkrylov[i] += yj * Zj[i];
    }
    // check the actual error on the xkrylov vector --> necessary since the
    // flexi preconditioner breaks the krylov subspace
    (field->*FunctionImage)(im, xkrylov);
    sub(r, b, im, xkrylovlen);
    double final_error = normP(r, xkrylovlen, &fieldcomm);
    // std::cout<< "Final error X: "<< final_error << std::endl;

    // if (initial_error <= rho_tol) {
    if (final_error <= rho_tol) {
      if (is_output_thread()) {
        printf(
            "FGMRES converged at restart # %d; iteration #%d with error: %g\n",
            itr, k, initial_error / rho_tol * tol);
        // cout << "GMRES converged at restart # " << itr << "; iteration #" <<
        // k << " with error: " << initial_error / rho_tol * tol << endl;
      }
      break;
    }
    if (is_output_thread() && GMRESVERBOSE) {
      printf("Restart: %d error: %g\n", itr, initial_error / rho_tol * tol);
      // cout << "Restart: " << itr << " error: " << initial_error / rho_tol *
      // tol << endl;
    }
  }
  if (itr == max_iter && is_output_thread()) {
    printf("FGMRES not converged !! Final error: %g\n",
           initial_error / rho_tol * tol);
    // cout << "GMRES not converged !! Final error: " << initial_error / rho_tol
    // * tol << endl;
  }

  delete[] r;
  delete[] im;
  delete[] s;
  delete[] cs;
  delete[] sn;
  delete[] y;
  delArr2(H, m + 1);
  delArr2(V, m + 1);
  delArr2(Z, m);

  delete[] rPrec;
  delete[] imPrec;
  delete[] sPrec;
  delete[] csPrec;
  delete[] snPrec;
  delete[] yPrec;
  delArr2(HPrec, mPrec + 1);
  delArr2(VPrec, mPrec + 1);
  return;
}

void GMRESasPreconditioner(FIELD_IMAGE FunctionImage, double* xkrylov,
                           int xkrylovlen, const double* b, int m, int max_iter,
                           double tol, EMfields3D* field, double* r, double* im,
                           double* s, double* cs, double* sn, double* y,
                           double** H, double** V) {
  if (m > xkrylovlen) {
    // m need not be the same for all processes,
    // we cannot restrict this test to the main process,
    // (although we could probably restrict it to the
    // process with the highest cartesian rank).
    eprintf("In GMRES the dimension of Krylov space(m) "
            "can't be > (length of krylov vector)/(# processors)\n");
  }
  bool GMRESVERBOSE = false;
  double initial_error, rho_tol;

  eqValue(0.0, s, m + 1);
  eqValue(0.0, cs, m + 1);
  eqValue(0.0, sn, m + 1);
  eqValue(0.0, y, m + 3);

  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < m; jj++)
      H[ii][jj] = 0;
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < xkrylovlen; jj++)
      V[ii][jj] = 0;

  MPI_Comm fieldcomm = (field->get_vct()).getFieldComm();

  double normb = normP(b, xkrylovlen, &fieldcomm);
  if (normb == 0.0)
    normb = 1.0;

  int itr = 0;
  for (itr = 0; itr < max_iter; itr++) {
    // r = b - A*x
    (field->*FunctionImage)(im, xkrylov);
    sub(r, b, im, xkrylovlen);
    initial_error = normP(r, xkrylovlen, &fieldcomm);

    if (itr == 0) {
      /*
      if (is_output_thread())
      printf("Initial residual prec: %g norm b vector (source) =
      %g\n",initial_error, normb);
      //cout << "Initial residual: " << initial_error << " norm b vector
      (source) = " << normb << endl;
      */
      rho_tol = initial_error * tol;

      if ((initial_error / normb) <= tol) {
        // if (is_output_thread())
        //   printf("GMRES preconditioner converged without iterations: initial
        //   error < tolerance\n");
        // cout << "GMRES converged without iterations: initial error <
        // tolerance" << endl;
        break;
      }
    }

    scale(V[0], r, (1.0 / initial_error), xkrylovlen);
    eqValue(0.0, s, m + 1);
    s[0] = initial_error;
    int k = 0;
    while (rho_tol < initial_error && k < m) {

      // w= A*V(:,k)
      double* w = V[k + 1];
      (field->*FunctionImage)(w, V[k]);
      // old code (many MPI_Allreduce calls)
      //
      // const double av = normP(w, xkrylovlen);
      // for (register int j = 0; j <= k; j++) {
      //  H[j][k] = dotP(w, V[j], xkrylovlen);
      //  addscale(-H[j][k], w, V[j], xkrylovlen);
      //}

      // new code to make a single MPI_Allreduce call
      for (int j = 0; j <= k; j++) {
        y[j] = dot(w, V[j], xkrylovlen);
      }
      y[k + 1] = norm2(w, xkrylovlen);

      MPI_Allreduce(MPI_IN_PLACE, y, (k + 2), MPI_DOUBLE, MPI_SUM, fieldcomm);

      for (int j = 0; j <= k; j++) {
        H[j][k] = y[j];
        addscale(-H[j][k], V[k + 1], V[j], xkrylovlen);
      }
      // Is there a numerically stable way to
      // eliminate this second all-reduce all?
      H[k + 1][k] = normP(V[k + 1], xkrylovlen, &fieldcomm);
      //
      // check that vectors are orthogonal
      //
      // for (register int j = 0; j <= k; j++) {
      //  dprint(dotP(w, V[j], xkrylovlen));
      //}

      double av = sqrt(y[k + 1]);
      // why are we testing floating point numbers
      // for equality?  Is this supposed to say
      // if (av < delta * fabs(H[k + 1][k]))
      const double delta = 0.001;
      if (av + delta * H[k + 1][k] == av) {
        for (int j = 0; j <= k; j++) {
          const double htmp = dotP(w, V[j], xkrylovlen, &fieldcomm);
          H[j][k] = H[j][k] + htmp;
          addscale(-htmp, w, V[j], xkrylovlen);
        }
        H[k + 1][k] = normP(w, xkrylovlen, &fieldcomm);
      }
      // normalize the new vector
      scale(w, (1.0 / H[k + 1][k]), xkrylovlen);

      if (0 < k) {
        for (int j = 0; j < k; j++)
          ApplyPlaneRotation(H[j + 1][k], H[j][k], cs[j], sn[j]);

        getColumn(y, H, k, m + 1);
      }

      const double mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k] = H[k][k] / mu;
      sn[k] = -H[k + 1][k] / mu;
      H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      ApplyPlaneRotation(s[k + 1], s[k], cs[k], sn[k]);
      initial_error = fabs(s[k]);
      k++;
    }

    k--;
    y[k] = s[k] / H[k][k];

    for (int i = k - 1; i >= 0; i--) {
      double tmp = 0.0;
      for (int l = i + 1; l <= k; l++)
        tmp += H[i][l] * y[l];
      y[i] = (s[i] - tmp) / H[i][i];
    }

    for (int j = 0; j < k; j++) {
      const double yj = y[j];
      double* Vj = V[j];
      for (int i = 0; i < xkrylovlen; i++)
        xkrylov[i] += yj * Vj[i];
    }

    if (initial_error <= rho_tol)
      break;
  }

  return;
}

void GMRESasPreconditionerNoComm(FIELD_IMAGE FunctionImage, double* xkrylov,
                                 int xkrylovlen, const double* b, int m,
                                 int max_iter, double tol, EMfields3D* field) {
  // if (is_output_thread())
  //   std::cout<< "Init GMRESasPreconditionerNoComm " <<std::endl;
  if (m > xkrylovlen) {
    // m need not be the same for all processes,
    // we cannot restrict this test to the main process,
    // (although we could probably restrict it to the
    // process with the highest cartesian rank).
    eprintf("In GMRES the dimension of Krylov space(m) "
            "can't be > (length of krylov vector)/(# processors)\n");
  }
  bool GMRESVERBOSE = false;
  double initial_error, rho_tol;
  double* r = new double[xkrylovlen];
  double* im = new double[xkrylovlen];

  double* s = new double[m + 1];
  double* cs = new double[m + 1];
  double* sn = new double[m + 1];
  double* y = new double[m + 3];
  eqValue(0.0, s, m + 1);
  eqValue(0.0, cs, m + 1);
  eqValue(0.0, sn, m + 1);
  eqValue(0.0, y, m + 3);

  // allocate H for storing the results from decomposition
  double** H = newArr2(double, m + 1, m);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < m; jj++)
      H[ii][jj] = 0;
  // allocate V
  double** V = newArr2(double, m + 1, xkrylovlen);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < xkrylovlen; jj++)
      V[ii][jj] = 0;

  if (GMRESVERBOSE && is_output_thread()) {
    printf("------------------------------------\n"
           "-             GMRES preconditioner No comm              -\n"
           "------------------------------------\n\n");
  }

  double normb = sqrt(norm2(b, xkrylovlen));
  if (normb == 0.0)
    normb = 1.0;

  int itr = 0;
  for (itr = 0; itr < max_iter; itr++) {
    // r = b - A*x
    (field->*FunctionImage)(im, xkrylov);
    sub(r, b, im, xkrylovlen);
    initial_error = sqrt(norm2(r, xkrylovlen));

    if (itr == 0) {
      if (GMRESVERBOSE && is_output_thread())
        printf(
            "Initial residual prec No Comm: %g norm b vector (source) = %g\n",
            initial_error, normb);
      // cout << "Initial residual: " << initial_error << " norm b vector
      // (source) = " << normb << endl;
      rho_tol = initial_error * tol;

      if ((initial_error / normb) <= tol) {
        if (GMRESVERBOSE && is_output_thread())
          printf("GMRES preconditioner No Comm converged without iterations: "
                 "initial error < tolerance\n");
        // cout << "GMRES converged without iterations: initial error <
        // tolerance" << endl;
        break;
      }
    }

    scale(V[0], r, (1.0 / initial_error), xkrylovlen);
    eqValue(0.0, s, m + 1);
    s[0] = initial_error;
    int k = 0;
    while (rho_tol < initial_error && k < m) {

      // w= A*V(:,k)
      double* w = V[k + 1];
      (field->*FunctionImage)(w, V[k]);
      // old code (many MPI_Allreduce calls)
      //
      // const double av = normP(w, xkrylovlen);
      // for (register int j = 0; j <= k; j++) {
      //  H[j][k] = dotP(w, V[j], xkrylovlen);
      //  addscale(-H[j][k], w, V[j], xkrylovlen);
      //}

      // new code to make a single MPI_Allreduce call
      for (int j = 0; j <= k; j++) {
        y[j] = dot(w, V[j], xkrylovlen);
      }
      y[k + 1] = norm2(w, xkrylovlen);
      {

        // MPI_Allreduce(MPI_IN_PLACE, y, (k+2),MPI_DOUBLE, MPI_SUM, fieldcomm);
      }
      for (int j = 0; j <= k; j++) {
        H[j][k] = y[j];
        addscale(-H[j][k], V[k + 1], V[j], xkrylovlen);
      }
      // Is there a numerically stable way to
      // eliminate this second all-reduce all?
      H[k + 1][k] = sqrt(norm2(V[k + 1], xkrylovlen));
      //
      // check that vectors are orthogonal
      //
      // for (register int j = 0; j <= k; j++) {
      //  dprint(dotP(w, V[j], xkrylovlen));
      //}

      double av = sqrt(y[k + 1]);
      // why are we testing floating point numbers
      // for equality?  Is this supposed to say
      // if (av < delta * fabs(H[k + 1][k]))
      const double delta = 0.001;
      if (av + delta * H[k + 1][k] == av) {
        for (int j = 0; j <= k; j++) {
          const double htmp = dot(w, V[j], xkrylovlen);
          H[j][k] = H[j][k] + htmp;
          addscale(-htmp, w, V[j], xkrylovlen);
        }
        H[k + 1][k] = sqrt(norm2(w, xkrylovlen));
      }

      // normalize the new vector
      scale(w, (1.0 / H[k + 1][k]), xkrylovlen);

      if (0 < k) {

        for (int j = 0; j < k; j++)
          ApplyPlaneRotation(H[j + 1][k], H[j][k], cs[j], sn[j]);

        getColumn(y, H, k, m + 1);
      }

      const double mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k] = H[k][k] / mu;
      sn[k] = -H[k + 1][k] / mu;
      H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      ApplyPlaneRotation(s[k + 1], s[k], cs[k], sn[k]);
      initial_error = fabs(s[k]);
      k++;
    }

    k--;
    y[k] = s[k] / H[k][k];

    for (int i = k - 1; i >= 0; i--) {
      double tmp = 0.0;
      for (int l = i + 1; l <= k; l++)
        tmp += H[i][l] * y[l];
      y[i] = (s[i] - tmp) / H[i][i];
    }

    for (int j = 0; j < k; j++) {
      const double yj = y[j];
      double* Vj = V[j];
      for (int i = 0; i < xkrylovlen; i++)
        xkrylov[i] += yj * Vj[i];
    }

    if (initial_error <= rho_tol) {
      if (is_output_thread()) {
        printf("GMRES preconditioner converged at restart # %d; iteration #%d "
               "with error: %g\n",
               itr, k, initial_error / rho_tol * tol);
        // cout << "GMRES converged at restart # " << itr << "; iteration #" <<
        // k << " with error: " << initial_error / rho_tol * tol << endl;
      }
      break;
    }
    if (is_output_thread() && GMRESVERBOSE) {
      printf("GMRES preconditioner Restart: %d error: %g\n", itr,
             initial_error / rho_tol * tol);
      // cout << "Restart: " << itr << " error: " << initial_error / rho_tol *
      // tol << endl;
    }
  }
  if (itr == max_iter && is_output_thread()) {
    printf("GMRES preconditioner not converged !! Final error: %g\n",
           initial_error / rho_tol * tol);
    // cout << "GMRES not converged !! Final error: " << initial_error / rho_tol
    // * tol << endl;
  }

  delete[] r;
  delete[] im;
  delete[] s;
  delete[] cs;
  delete[] sn;
  delete[] y;
  delArr2(H, m + 1);
  delArr2(V, m + 1);
  return;
}

void GMRES(FIELD_IMAGE FunctionImage, double* xkrylov, int xkrylovlen,
           const double* b, int m, int max_iter, double tol, Field* field) {
  if (m > xkrylovlen) {
    // m need not be the same for all processes,
    // we cannot restrict this test to the main process,
    // (although we could probably restrict it to the
    // process with the highest cartesian rank).
    eprintf("In GMRES the dimension of Krylov space(m) "
            "can't be > (length of krylov vector)/(# processors)\n");
  }
  bool GMRESVERBOSE = false;
  double initial_error, rho_tol;
  double* r = new double[xkrylovlen];
  double* im = new double[xkrylovlen];

  double* s = new double[m + 1];
  double* cs = new double[m + 1];
  double* sn = new double[m + 1];
  double* y = new double[m + 3];
  eqValue(0.0, s, m + 1);
  eqValue(0.0, cs, m + 1);
  eqValue(0.0, sn, m + 1);
  eqValue(0.0, y, m + 3);

  // allocate H for storing the results from decomposition
  double** H = newArr2(double, m + 1, m);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < m; jj++)
      H[ii][jj] = 0;
  // allocate V
  double** V = newArr2(double, m + 1, xkrylovlen);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < xkrylovlen; jj++)
      V[ii][jj] = 0;

  if (GMRESVERBOSE && is_output_thread()) {
    printf("------------------------------------\n"
           "-             GMRES                -\n"
           "------------------------------------\n\n");
  }

  MPI_Comm fieldcomm = (field->get_vct()).getFieldComm();

  double normb = normP(b, xkrylovlen, &fieldcomm);
  if (normb == 0.0)
    normb = 1.0;

  int itr = 0;
  for (itr = 0; itr < max_iter; itr++) {
    // r = b - A*x
    (field->*FunctionImage)(im, xkrylov);
    sub(r, b, im, xkrylovlen);
    initial_error = normP(r, xkrylovlen, &fieldcomm);

    if (itr == 0) {
      if (is_output_thread())
        printf("Initial residual: %g norm b vector (source) = %g\n",
               initial_error, normb);
      // cout << "Initial residual: " << initial_error << " norm b vector
      // (source) = " << normb << endl;
      rho_tol = initial_error * tol;

      if ((initial_error / normb) <= tol) {
        if (is_output_thread())
          printf("GMRES converged without iterations: initial error < "
                 "tolerance\n");
        // cout << "GMRES converged without iterations: initial error <
        // tolerance" << endl;
        break;
      }
    }

    scale(V[0], r, (1.0 / initial_error), xkrylovlen);
    eqValue(0.0, s, m + 1);
    s[0] = initial_error;
    int k = 0;
    while (rho_tol < initial_error && k < m) {

      // w= A*V(:,k)
      double* w = V[k + 1];
      (field->*FunctionImage)(w, V[k]);
      // old code (many MPI_Allreduce calls)
      //
      // const double av = normP(w, xkrylovlen);
      // for (register int j = 0; j <= k; j++) {
      //  H[j][k] = dotP(w, V[j], xkrylovlen);
      //  addscale(-H[j][k], w, V[j], xkrylovlen);
      //}

      // new code to make a single MPI_Allreduce call
      for (int j = 0; j <= k; j++) {
        y[j] = dot(w, V[j], xkrylovlen);
      }
      y[k + 1] = norm2(w, xkrylovlen);
      {

        MPI_Allreduce(MPI_IN_PLACE, y, (k + 2), MPI_DOUBLE, MPI_SUM, fieldcomm);
      }
      for (int j = 0; j <= k; j++) {
        H[j][k] = y[j];
        addscale(-H[j][k], V[k + 1], V[j], xkrylovlen);
      }
      // Is there a numerically stable way to
      // eliminate this second all-reduce all?
      H[k + 1][k] = normP(V[k + 1], xkrylovlen, &fieldcomm);
      //
      // check that vectors are orthogonal
      //
      // for (register int j = 0; j <= k; j++) {
      //  dprint(dotP(w, V[j], xkrylovlen));
      //}

      double av = sqrt(y[k + 1]);
      // why are we testing floating point numbers
      // for equality?  Is this supposed to say
      // if (av < delta * fabs(H[k + 1][k]))
      const double delta = 0.001;
      if (av + delta * H[k + 1][k] == av) {
        for (int j = 0; j <= k; j++) {
          const double htmp = dotP(w, V[j], xkrylovlen, &fieldcomm);
          H[j][k] = H[j][k] + htmp;
          addscale(-htmp, w, V[j], xkrylovlen);
        }
        H[k + 1][k] = normP(w, xkrylovlen, &fieldcomm);
      }
      // normalize the new vector
      scale(w, (1.0 / H[k + 1][k]), xkrylovlen);

      if (0 < k) {

        for (int j = 0; j < k; j++)
          ApplyPlaneRotation(H[j + 1][k], H[j][k], cs[j], sn[j]);

        getColumn(y, H, k, m + 1);
      }

      const double mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k] = H[k][k] / mu;
      sn[k] = -H[k + 1][k] / mu;
      H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      ApplyPlaneRotation(s[k + 1], s[k], cs[k], sn[k]);
      initial_error = fabs(s[k]);
      k++;
    }

    k--;
    y[k] = s[k] / H[k][k];

    for (int i = k - 1; i >= 0; i--) {
      double tmp = 0.0;
      for (int l = i + 1; l <= k; l++)
        tmp += H[i][l] * y[l];
      y[i] = (s[i] - tmp) / H[i][i];
    }

    for (int j = 0; j < k; j++) {
      const double yj = y[j];
      double* Vj = V[j];
      for (int i = 0; i < xkrylovlen; i++)
        xkrylov[i] += yj * Vj[i];
    }

    if (initial_error <= rho_tol) {
      if (is_output_thread()) {
        printf(
            "GMRES converged at restart # %d; iteration #%d with error: %g\n",
            itr, k, initial_error / rho_tol * tol);
        // cout << "GMRES converged at restart # " << itr << "; iteration #" <<
        // k << " with error: " << initial_error / rho_tol * tol << endl;
      }
      break;
    }
    if (is_output_thread() && GMRESVERBOSE) {
      printf("Restart: %d error: %g\n", itr, initial_error / rho_tol * tol);
      // cout << "Restart: " << itr << " error: " << initial_error / rho_tol *
      // tol << endl;
    }
  }
  if (itr == max_iter && is_output_thread()) {
    printf("GMRES not converged !! Final error: %g\n",
           initial_error / rho_tol * tol);
    // cout << "GMRES not converged !! Final error: " << initial_error / rho_tol
    // * tol << endl;
  }

  delete[] r;
  delete[] im;
  delete[] s;
  delete[] cs;
  delete[] sn;
  delete[] y;
  delArr2(H, m + 1);
  delArr2(V, m + 1);
  return;
}

void ApplyPlaneRotation(double& dx, double& dy, double& cs, double& sn) {
  double temp = cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = temp;
}
