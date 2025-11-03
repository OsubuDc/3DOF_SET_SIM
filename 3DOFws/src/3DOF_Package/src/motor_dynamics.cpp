#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>

using namespace std;

class BLDCMotor {
private:
    // Motor parameters (example values - replace with your ECXF 42M datasheet values)
    double R;       // Terminal resistance [Ohm]
    double L;       // Terminal inductance [H]
    double K_t;     // Torque constant [Nm/A]
    double K_e;     // Back-EMF constant [V·s/rad]
    double J;       // Rotor inertia [kg·m²]
    double B;       // Viscous friction [Nm·s/rad]
    
    // State vector: x = [i, ω]^T
    // i  = current [A]
    // ω  = angular velocity [rad/s]
    vector<double> x;
    
    // State-space matrices
    vector<vector<double>> A;  // 2x2 system matrix
    vector<double> B;           // 2x1 input matrix
    
public:
    BLDCMotor(double resistance, double inductance, double torque_const, 
              double back_emf_const, double inertia, double friction) 
        : R(resistance), L(inductance), K_t(torque_const), 
          K_e(back_emf_const), J(inertia), B(friction) {
        
        // Initialize state vector [current, velocity]
        x = {0.0, 0.0};
        
        // Initialize matrices
        A.resize(2, vector<double>(2));
        B.resize(2);
        
        computeStateSpaceMatrices();
    }
    
    void computeStateSpaceMatrices() {
        /*
         * State-space model derivation:
         * 
         * Electrical equation: L·di/dt = V - R·i - K_e·ω
         * Mechanical equation: J·dω/dt = K_t·i - B·ω - T_load
         * 
         * State vector: x = [i, ω]^T
         * Input: u = [V, T_load]^T (we'll use V only for simplicity)
         * 
         * dx/dt = A·x + B·u
         * 
         * [di/dt]   [-R/L    -K_e/L] [i]   [1/L  0] [V     ]
         * [dω/dt] = [K_t/J   -B/J  ] [ω] + [0   -1/J] [T_load]
         * 
         * For single input (voltage only, no load):
         * B = [1/L, 0]^T
         */
        
        // Matrix A
        A[0][0] = -R / L;           // di/dt dependency on i
        A[0][1] = -K_e / L;         // di/dt dependency on ω
        A[1][0] = K_t / J;          // dω/dt dependency on i
        A[1][1] = -B / J;           // dω/dt dependency on ω
        
        // Matrix B (for voltage input)
        B[0] = 1.0 / L;
        B[1] = 0.0;
    }
    
    void computeEigenvalues() {
        /*
         * For 2x2 matrix: A = [a  b]
         *                     [c  d]
         * 
         * Characteristic equation: det(A - λI) = 0
         * λ² - (a+d)λ + (ad-bc) = 0
         * 
         * λ = [(a+d) ± sqrt((a+d)² - 4(ad-bc))] / 2
         */
        
        double a = A[0][0];
        double b = A[0][1];
        double c = A[1][0];
        double d = A[1][1];
        
        double trace = a + d;
        double det = a*d - b*c;
        double discriminant = trace*trace - 4*det;
        
        cout << "\n=== EIGENVALUE ANALYSIS ===" << endl;
        cout << "Trace(A) = " << trace << endl;
        cout << "Det(A) = " << det << endl;
        cout << "Discriminant = " << discriminant << endl;
        
        if (discriminant >= 0) {
            // Real eigenvalues
            double lambda1 = (trace + sqrt(discriminant)) / 2.0;
            double lambda2 = (trace - sqrt(discriminant)) / 2.0;
            
            cout << "\nEigenvalues (real):" << endl;
            cout << "λ₁ = " << lambda1 << endl;
            cout << "λ₂ = " << lambda2 << endl;
            
            // System stability analysis
            if (lambda1 < 0 && lambda2 < 0) {
                cout << "\n✓ System is STABLE (both eigenvalues negative)" << endl;
                cout << "  Dominant time constant τ = " << -1.0/max(lambda1, lambda2) << " seconds" << endl;
            } else {
                cout << "\n✗ System is UNSTABLE" << endl;
            }
        } else {
            // Complex conjugate eigenvalues
            double real_part = trace / 2.0;
            double imag_part = sqrt(-discriminant) / 2.0;
            
            cout << "\nEigenvalues (complex conjugate):" << endl;
            cout << "λ₁ = " << real_part << " + " << imag_part << "i" << endl;
            cout << "λ₂ = " << real_part << " - " << imag_part << "i" << endl;
            
            double natural_freq = sqrt(imag_part*imag_part + real_part*real_part);
            double damping_ratio = -real_part / natural_freq;
            
            cout << "\nNatural frequency ωₙ = " << natural_freq << " rad/s" << endl;
            cout << "Damping ratio ζ = " << damping_ratio << endl;
            
            if (real_part < 0) {
                cout << "\n✓ System is STABLE (negative real part)" << endl;
                if (damping_ratio < 1.0) {
                    cout << "  System is UNDERDAMPED (oscillatory response)" << endl;
                }
            } else {
                cout << "\n✗ System is UNSTABLE" << endl;
            }
        }
    }
    
    void printStateSpaceModel() {
        cout << fixed << setprecision(6);
        cout << "\n=== STATE-SPACE MODEL ===" << endl;
        cout << "State vector: x = [i, ω]^T" << endl;
        cout << "Input: u = V (voltage)" << endl;
        cout << "\ndx/dt = A·x + B·u\n" << endl;
        
        cout << "Matrix A:" << endl;
        cout << "[" << setw(12) << A[0][0] << " " << setw(12) << A[0][1] << "]" << endl;
        cout << "[" << setw(12) << A[1][0] << " " << setw(12) << A[1][1] << "]" << endl;
        
        cout << "\nMatrix B:" << endl;
        cout << "[" << B[0] << "]" << endl;
        cout << "[" << B[1] << "]" << endl;
        
        cout << "\n=== PHYSICAL INTERPRETATION ===" << endl;
        cout << "A[0][0] = -R/L = electrical time constant effect" << endl;
        cout << "A[0][1] = -K_e/L = back-EMF coupling" << endl;
        cout << "A[1][0] = K_t/J = torque production" << endl;
        cout << "A[1][1] = -B/J = mechanical damping" << endl;
    }
    
    vector<double> simulate_step(double V, double dt) {
        // Euler integration: x(k+1) = x(k) + dt·(A·x(k) + B·u(k))
        vector<double> dx(2);
        
        // Compute dx = A·x + B·u
        dx[0] = A[0][0]*x[0] + A[0][1]*x[1] + B[0]*V;
        dx[1] = A[1][0]*x[0] + A[1][1]*x[1] + B[1]*V;
        
        // Update state
        x[0] += dx[0] * dt;
        x[1] += dx[1] * dt;
        
        return x;
    }
    
    void printState() {
        cout << "Current i = " << x[0] << " A, ";
        cout << "Velocity ω = " << x[1] << " rad/s (";
        cout << x[1]*60.0/(2*M_PI) << " RPM)" << endl;
    }
};

int main() {
    cout << "\n╔════════════════════════════════════════════╗" << endl;
    cout << "║  BLDC Motor State-Space Model & Analysis  ║" << endl;
    cout << "║  Maxon ECXF 42M (48V)                     ║" << endl;
    cout << "╚════════════════════════════════════════════╝\n" << endl;
    
    // Example parameters (REPLACE with actual ECXF 42M datasheet values!)
    double R = 0.5;         // Terminal resistance [Ohm]
    double L = 0.001;       // Terminal inductance [H] = 1 mH
    double K_t = 0.05;      // Torque constant [Nm/A]
    double K_e = 0.05;      // Back-EMF constant [V·s/rad]
    double J = 0.0001;      // Rotor inertia [kg·m²]
    double B = 0.00001;     // Viscous friction [Nm·s/rad]
    
    cout << "=== MOTOR PARAMETERS ===" << endl;
    cout << "R  = " << R << " Ω" << endl;
    cout << "L  = " << L*1000 << " mH" << endl;
    cout << "K_t = " << K_t << " Nm/A" << endl;
    cout << "K_e = " << K_e << " V·s/rad" << endl;
    cout << "J  = " << J << " kg·m²" << endl;
    cout << "B  = " << B << " Nm·s/rad" << endl;
    
    BLDCMotor motor(R, L, K_t, K_e, J, B);
    
    motor.printStateSpaceModel();
    motor.computeEigenvalues();
    
    // Simple simulation example
    cout << "\n=== STEP RESPONSE SIMULATION ===" << endl;
    cout << "Applying 48V step input...\n" << endl;
    
    double V = 48.0;  // Voltage input
    double dt = 0.0001;  // Time step [s]
    int steps = 5;
    
    for (int i = 0; i <= steps; i++) {
        double t = i * dt * 1000;  // Convert to ms
        cout << "t = " << t << " ms: ";
        motor.printState();
        motor.simulate_step(V, dt);
    }
    
    cout << "\n=== REFERENCES ===" << endl;
    cout << "1. State-space modeling: 'Modern Control Engineering' - Ogata" << endl;
    cout << "2. Motor equations: 'Electric Machinery Fundamentals' - Chapman" << endl;
    cout << "3. Eigenvalue stability: 'Linear System Theory' - Chen" << endl;
    
    return 0;
}