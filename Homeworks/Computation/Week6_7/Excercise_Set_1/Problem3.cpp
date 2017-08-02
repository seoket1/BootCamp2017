#include <iostream>
#include <cmath>
using namespace std;

/* I refered "https://www.programiz.com/cpp-programming/examples/quadratic-roots"*/

int main() {
    float a, b, c, x, y, determinant, real, imag;
    cout << "Type coefficients a, b and c of quadratic equation: ";
    cin >> a >> b >> c;
    determinant = pow(b, 2) - 4*a*c;
    
    if (determinant > 0) {
        x = (-b + sqrt(determinant)) / (2*a);
        y = (-b - sqrt(determinant)) / (2*a);
        cout << "x = " << x << endl;
        cout << "y = " << y << endl;
    }
    
    else if (determinant == 0) {
        x = (-b + sqrt(determinant)) / (2*a);
        cout << "x = y =" << x << endl;
    }

    else {
        real = -b/(2*a);
        imag =sqrt(-determinant)/(2*a);
        cout << "x = " << real << "+" << imag << "i" << endl;
        cout << "y = " << real << "-" << imag << "i" << endl;
    }

    return 0;
}
