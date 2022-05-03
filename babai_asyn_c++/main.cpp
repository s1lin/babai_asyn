//#include "src/example/cils_LLL_test.cpp"
//#include "src/example/cils_PBNP_test.cpp"
#include "src/example/cils_PBOB_test.cpp"

using namespace std;
using namespace cils;


int main(int argc, char *argv[]) {


    printf("\n====================[ Run | cils | Release ]==================================\n");
    double t = omp_get_wtime();

    int size_n = stoi(argv[1]);
    int is_local = stoi(argv[2]);
////    CILS cils = cils_driver<double, int>(argc, argv);
//    plot_LLL<double, int>();

    test_PBOB<double, int>(size_n, is_local);

    t = omp_get_wtime() - t;

    printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
           t, t / 60, t / 3600);


    return 0;
}
