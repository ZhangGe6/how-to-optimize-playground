
void MMult_base(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
void MMult_unroll(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);
void MMult_unroll_inner(int m, int k, int n, double *A, double *B, double *C, int lda, int ldb, int ldc);