#include <iostream>
#include <math.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;

int main (){
	float n, count, x, y, pi, i;
	n = 10000000;
	count = 0;
	for (i=0; i <= n; i++){
		x = (float)rand()/RAND_MAX;
		y = (float)rand()/RAND_MAX;
	    //cout << "x = " << x << endl;
	    //cout << "y = " << y << endl;

	    if (x*x + y*y<= 1.0){
	    	count++;
		}
	}
	cout << "count = " << count << endl;
	cout << "n = " << n << endl;
	pi = (count / n) * 4   ;
	cout << "pi = " << pi << endl;
	return 0;  
}
  

