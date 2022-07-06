//#include "src/example/cils_LLL_test.cpp"
//#include "src/example/cils_PBNP_test.cpp"
//#include "src/example/cils_PBOB_test.cpp"
#include "src/example/cils_PBSIC_test.cpp"

using namespace std;
using namespace cils;

void functiona(int i, int n_threads) {
    cout << i << "," << omp_get_thread_num() << endl;
//#pragma omp parallel num_threads(n_threads)
//    for (int t = 0; t < 2; t++)
//#pragma omp for nowait
//            for (int j = 0; j < 4; j++)
//                printf("Task %d: thread %d of the %d children of %d: handling iter %d\n",
//                       i, omp_get_thread_num(), omp_get_team_size(2),
//                       omp_get_ancestor_thread_num(1), j);
}

int main(int argc, char *argv[]) {


    printf("\n====================[ Run | cils | Release ]==================================\n");
    double t = omp_get_wtime();

    int size_n = stoi(argv[1]);
//    int nob = stoi(argv[2]);
//    int c = stoi(argv[3]);
//    int T = stoi(argv[4]);
    int is_local = stoi(argv[2]);
    int info = stoi(argv[3]);
    int sec = stoi(argv[4]);

//    test_PBOB<double, int>(size_n, nob, c, T, is_local);
//    test_init<double, int>(size_n, 1);
    test_pbsic<double, int>(size_n, is_local, info, 56, sec);
//    test_init_pt<double, int>();

//    omp_set_nested(1);   /* make sure nested parallism is on */
//    int nprocs = omp_get_num_procs();
//    auto nthreads = new int[2]{4, 4};

//#pragma omp parallel default(none) shared(nthreads) num_threads(4)
//    {
//#pragma omp single
//        {
//            for (int i = 0; i < 6; i++)
//#pragma omp task
//                functiona(i, nthreads[i]);
//        }
//
//    }

    t = omp_get_wtime() - t;

    printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
           t, t / 60, t / 3600);


    return 0;
}
