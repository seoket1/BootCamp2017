#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>
 
int main()
{
    double niter = 100000000;
    double x,y;
    int i;
    int count=0;
    double z;
    double pi;
    //srand(time(NULL));
    //main loop
    
    #pragma omp parallel for
    for (i=0; i<niter; ++i)
    {
        //get random points
        x = (double)rand()/RAND_MAX;
        y = (double)rand()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    pi = ((double)count/(double)niter)*4.0;          //p = 4(m/n)
    printf("Pi: %f\n", pi);
}
