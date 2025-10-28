// pendulum_simulator.cpp
// ROS2 C++ node for simulating a 2-DOF robotic arm with operational space control

// ============================================================================
// INCLUDES - External libraries and ROS2 components
// ============================================================================

// Core ROS2 functionality
#include "rclcpp/rclcpp.hpp"              // Main ROS2 C++ API: Node class, logging, timers
#include "sensor_msgs/msg/joint_state.hpp" // Message type for publishing joint data (position, velocity, effort)
#include "std_msgs/msg/header.hpp"        // Message header: timestamp and frame_id

// Pinocchio - Rigid body dynamics library
#include <pinocchio/fwd.hpp>              // Forward declarations (must be first)
#include <pinocchio/parsers/urdf.hpp>     // URDF parser to load robot model
#include <pinocchio/algorithm/kinematics.hpp>        // Forward kinematics
#include <pinocchio/algorithm/jacobian.hpp>          // Jacobian computation
#include <pinocchio/algorithm/frames.hpp>            // Frame transformations
#include <pinocchio/algorithm/joint-configuration.hpp> // Joint limit handling
#include <pinocchio/algorithm/crba.hpp>              // Composite Rigid Body Algorithm (mass matrix)
#include <pinocchio/algorithm/rnea.hpp>              // Recursive Newton-Euler Algorithm (dynamics)

// Standard C++ libraries
#include <Eigen/Dense>                    // Linear algebra: vectors, matrices, solvers
#include <memory>                         // Smart pointers: std::shared_ptr
#include <vector>                         // Dynamic arrays
#include <string>                         // String handling
#include <cmath>                          // Math functions (M_PI)
#include <chrono>                         // Time durations and time points

// Boost for numerical integration
#include <boost/numeric/odeint.hpp>       // ODE solver (equivalent to scipy.solve_ivp)

// ============================================================================
// TYPE ALIASES - Shorthand for commonly used types
// ============================================================================
using namespace std::chrono_literals;     // Enables time literals like 10ms, 1s

// Pinocchio namespace shortcuts
namespace pin = pinocchio;                // pin::Model instead of pinocchio::Model

// State vector type: [q1, q2, v1, v2] (positions and velocities)
typedef Eigen::VectorXd state_type;

// ============================================================================
// CLASS DEFINITION - ROS2 Node for Pendulum Simulation
// ============================================================================

// Inherits from rclcpp::Node, gaining all ROS2 node capabilities:
// - Parameter management (declare/get parameters)
// - Publishers/subscribers for communication
// - Timers for periodic execution
// - Logging (RCLCPP_INFO, RCLCPP_WARN, etc.)
class PendulumSimulator : public rclcpp::Node
{
public:
    // ------------------------------------------------------------------------
    // CONSTRUCTOR - Initializes the node and all its components
    // ------------------------------------------------------------------------
    PendulumSimulator() 
        : Node("pendulum_simulator"),  // Initialize parent class with node name
          time_(0.0),                  // Start simulation time at zero
          dt_(0.01)                    // Timestep: 100 Hz (same as Python version)
    {
        // ====================================================================
        // STEP 1: DECLARE PARAMETERS
        // ====================================================================
        // ROS2 parameters allow external configuration without recompiling
        // Two-step process:
        //   1. declare_parameter() - Registers parameter with default value
        //   2. get_parameter() - Retrieves actual value (could be overridden)
        //
        // Usage from terminal:
        //   ros2 param list                              # See all parameters
        //   ros2 param get /pendulum_simulator urdf_path # Check value
        //   ros2 run pkg node --ros-args -p urdf_path:=/new/path/file.urdf
        
        this->declare_parameter<std::string>("urdf_path", 
            "/ThesisRosGITV1/src/urdf_2dof/urdf/2dof.urdf");
        this->declare_parameter<std::string>("mesh_dir", 
            "/ThesisRosGITV1/src/urdf_2dof/meshes");
        
        // Retrieve parameter values into local variables
        std::string urdf_path = this->get_parameter("urdf_path").as_string();
        std::string mesh_dir = this->get_parameter("mesh_dir").as_string();
        
        RCLCPP_INFO(this->get_logger(), "Loading URDF from: %s", urdf_path.c_str());
        
        // ====================================================================
        // STEP 2: LOAD ROBOT MODEL USING PINOCCHIO
        // ====================================================================
        // Pinocchio uses three main objects:
        //   - Model: Kinematic/dynamic structure (links, joints, inertias)
        //   - Data: Workspace for computations (preallocated memory)
        //   - GeometryModel: Collision/visual meshes (optional, not used here)
        
        std::vector<std::string> mesh_dirs = {mesh_dir};
        
        // Parse URDF file and build kinematic tree
        // This populates 'model_' with:
        //   - Joint types, axes, limits
        //   - Link masses, inertias, COM positions
        //   - Parent-child relationships
        pin::urdf::buildModel(urdf_path, model_);
        
        // Create data structure (allocates memory for intermediate calculations)
        data_ = pin::Data(model_);
        
        // Set gravity vector in world frame (Y-up convention)
        // model_.gravity is a Motion object with linear and angular parts
        model_.gravity.linear(Eigen::Vector3d(0.0, -9.81, 0.0));
        
        // ====================================================================
        // STEP 3: EXTRACT JOINT INFORMATION
        // ====================================================================
        // model_.names is std::vector<std::string> containing:
        //   Index 0: "universe" (world frame, not a real joint)
        //   Index 1+: Actual joint names from URDF
        // We skip index 0 to get only movable joints
        
        for (size_t i = 1; i < model_.names.size(); ++i) {
            joint_names_.push_back(model_.names[i]);
        }
        
        RCLCPP_INFO(this->get_logger(), "Found %zu joints:", joint_names_.size());
        for (const auto& name : joint_names_) {
            RCLCPP_INFO(this->get_logger(), "  - %s", name.c_str());
        }
        
        // ====================================================================
        // STEP 4: LOCATE END-EFFECTOR FRAME
        // ====================================================================
        // Frames are reference points attached to links (defined in URDF)
        // We need the end-effector frame to control its position
        
        std::string ee_frame_name = "EndEffector";
        
        // Search for frame by name (returns true if found)
        if (model_.existFrame(ee_frame_name)) {
            // getFrameId() returns numerical index for fast lookup
            ee_frame_id_ = model_.getFrameId(ee_frame_name);
            RCLCPP_INFO(this->get_logger(), "Found end-effector frame: %s (ID: %ld)", 
                        ee_frame_name.c_str(), ee_frame_id_);
        } else {
            // Fallback: use last frame if EndEffector not found
            RCLCPP_ERROR(this->get_logger(), "Could not find frame '%s'", ee_frame_name.c_str());
            RCLCPP_INFO(this->get_logger(), "Available frames:");
            for (size_t i = 0; i < model_.frames.size(); ++i) {
                RCLCPP_INFO(this->get_logger(), "  %zu: %s", i, model_.frames[i].name.c_str());
            }
            ee_frame_id_ = model_.frames.size() - 1;
            RCLCPP_WARN(this->get_logger(), "Using fallback frame: %s", 
                        model_.frames[ee_frame_id_].name.c_str());
        }
        
        // ====================================================================
        // STEP 5: CONTROL PARAMETERS
        // ====================================================================
        
        // Target end-effector POSE (position + orientation)
        // Position: (x, y, z) in meters
        // Orientation: Roll angle around X-axis (radians)
        x_des_ = Eigen::Vector3d(0.3, 0.3, 0.05);
        roll_des_ = 0.0;  // Desired roll angle (0 = neutral orientation)
        
        // Target velocity and acceleration (zero for stationary target)
        xdot_des_ = Eigen::Vector4d::Zero();    // [vx, vy, vz, omega_x]
        xddot_des_ = Eigen::Vector4d::Zero();   // [ax, ay, az, alpha_x]
        
        // PD gains for operational space control
        // Now 4×4 diagonal matrix for X, Y, Z, Roll control
        // Note: Roll gain might need different tuning than position gains
        Kp_ = Eigen::Matrix4d::Identity();
        Kp_(0,0) = 50.0;  // X position gain
        Kp_(1,1) = 50.0;  // Y position gain
        Kp_(2,2) = 50.0;  // Z position gain
        Kp_(3,3) = 30.0;  // Roll orientation gain (typically lower than position)
        
        Kd_ = Eigen::Matrix4d::Identity();
        Kd_(0,0) = 40.0;  // X velocity damping
        Kd_(1,1) = 40.0;  // Y velocity damping
        Kd_(2,2) = 40.0;  // Z velocity damping
        Kd_(3,3) = 25.0;  // Roll velocity damping
        
        // Joint limits (radians) - CRITICAL to prevent uncontrolled spinning
        // Joint 0: X-axis roll (arm rotation to avoid elbow flip)
        // Joint 1: Z-axis shoulder (elevation)
        // Joint 2: Z-axis elbow (flexion) - LIMITED RANGE prevents singularity!
        //
        // NOTE: Your URDF has Joint2 limits as lower=-0.1, upper=-2.6
        //       This appears inverted but may be intentional based on joint direction
        //       For simulation, we'll use symmetric limits. Adjust if needed:
        q_min_ = Eigen::Vector3d(-M_PI, -1.57, -2.6);   // [±180°, ±90°, -149°]
        q_max_ = Eigen::Vector3d(M_PI, 1.57, -0.1);     // Match URDF limits
        
        RCLCPP_INFO(this->get_logger(), "Target pose: pos=[%.3f, %.3f, %.3f] roll=%.1f°", 
                    x_des_[0], x_des_[1], x_des_[2], roll_des_ * 180.0/M_PI);
        RCLCPP_WARN(this->get_logger(), "Joint limits: [%.0f°, %.0f°, %.0f°] to [%.0f°, %.0f°, %.0f°]",
                    q_min_[0] * 180.0/M_PI, q_min_[1] * 180.0/M_PI, q_min_[2] * 180.0/M_PI,
                    q_max_[0] * 180.0/M_PI, q_max_[1] * 180.0/M_PI, q_max_[2] * 180.0/M_PI);
        
        // ====================================================================
        // STEP 6: INITIALIZE STATE
        // ====================================================================
        // State vector: [q1, q2, q3, v1, v2, v3]
        //   q1, q2, q3: Joint angles (radians)
        //   v1, v2, v3: Joint velocities (rad/s)
        
        Eigen::Vector3d q_init(0.01, 0.01, 0.01);  // Small non-zero initial angles
        Eigen::Vector3d v_init = Eigen::Vector3d::Zero();  // Start at rest
        
        // Resize state vector to 6 elements (3 positions + 3 velocities)
        state_.resize(model_.nq + model_.nv);  // nq = num positions, nv = num velocities
        state_ << q_init, v_init;  // Combine into single vector
        
        RCLCPP_INFO(this->get_logger(), "Initial configuration: [%.1f°, %.1f°, %.1f°]",
                    q_init[0] * 180.0/M_PI, q_init[1] * 180.0/M_PI, q_init[2] * 180.0/M_PI);
        
        // ====================================================================
        // STEP 7: CREATE ROS2 PUBLISHER
        // ====================================================================
        // Publishers broadcast messages on topics
        // Other nodes (like robot_state_publisher, RViz) subscribe to receive data
        //
        // Topic: /joint_states
        // Type: sensor_msgs/msg/JointState
        //   - header (timestamp, frame_id)
        //   - name (joint names)
        //   - position, velocity, effort (arrays matching names)
        // Queue size: 10 (buffer for 10 messages if subscriber is slow)
        
        joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "joint_states", 10);
        
        // ====================================================================
        // STEP 8: CREATE TIMER FOR PERIODIC EXECUTION
        // ====================================================================
        // Timers call a function at regular intervals (non-blocking)
        // Here: calls step_and_publish() every 10ms (100 Hz)
        //
        // How it works:
        //   - Timer is managed by ROS2 executor (rclcpp::spin)
        //   - Executor checks timer deadlines in background
        //   - When deadline reached, callback is invoked
        //   - Callback returns immediately, timer resets for next cycle
        
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(dt_),  // Convert dt_ to chrono duration
            std::bind(&PendulumSimulator::step_and_publish, this)  // Bind member function
        );
        
        RCLCPP_INFO(this->get_logger(), "Pendulum simulator started! Publishing at %.0f Hz", 1.0/dt_);
    }

private:
    // ------------------------------------------------------------------------
    // SYSTEM DYNAMICS - Computes ẋ = f(x, t) for ODE integrator
    // ------------------------------------------------------------------------
    // This is the "differential equation" that describes how the system evolves
    // Input:  state = [q1, q2, v1, v2], time t
    // Output: dstate/dt = [v1, v2, a1, a2] where a = qddot (acceleration)
    //
    // Called repeatedly by Boost.OdeInt during numerical integration
    
    void robot_dynamics(const state_type &state, state_type &dstate, double t)
    {
        // Extract positions and velocities from state vector
        Eigen::VectorXd q = state.head(model_.nq);  // First nq elements
        Eigen::VectorXd v = state.tail(model_.nv);  // Last nv elements
        
        // ====================================================================
        // FORWARD KINEMATICS - Compute link positions/orientations
        // ====================================================================
        // Updates data_.oMi (joint placements) and data_.oMf (frame placements)
        // oMi[i] = Transformation matrix from world origin to joint i
        // oMf[i] = Transformation matrix from world origin to frame i
        //
        // Why pass velocity?
        //   - Needed for Jacobian time derivative (J_dot)
        //   - Updates velocity-dependent terms in data_ structure
        
        pin::forwardKinematics(model_, data_, q, v);
        
        // Update frame placements (end-effector position)
        // This computes transforms for all frames defined in URDF
        pin::updateFramePlacements(model_, data_);
        
        // ====================================================================
        // JACOBIAN COMPUTATION - Maps joint velocities to EE velocities
        // ====================================================================
        // Jacobian: J(q) such that ẋ = J(q)q̇
        // Columns: How each joint affects end-effector motion
        // Rows: Effect on each Cartesian direction (x, y, z, rx, ry, rz)
        //
        // For 3-DOF robot with base rotation + shoulder + elbow:
        //   - Joint 1 (base): Rotates entire arm around Z-axis, affects X and Y
        //   - Joint 2 (shoulder): Lifts arm up/down, affects all X, Y, Z
        //   - Joint 3 (elbow): Extends/retracts forearm, affects X, Y, Z
        //
        // We control full 3D translational position (X, Y, Z)
        
        // Full 6×3 Jacobian (6 DOF: 3 linear + 3 angular, 3 joints)
        Eigen::MatrixXd J_full = pin::computeFrameJacobian(
            model_, data_, q, ee_frame_id_, pin::LOCAL_WORLD_ALIGNED);
        
        // Extract 4×3 block: X-Y-Z linear velocities + X-axis angular velocity (roll)
        // For 3-DOF robot controlling 3D position + roll orientation
        // J is 4×3 (4 task DOF, 3 joint DOF)
        // System is REDUNDANT (underdetermined) - infinite solutions exist
        // Use pseudoinverse to find minimum-norm solution (smoothest joint motion)
        Eigen::MatrixXd J(4, 3);
        J.topRows(3) = J_full.topRows(3);      // Linear velocities (X, Y, Z)
        J.row(3) = J_full.row(3);              // Angular velocity around X (roll)
        
        // Time derivative of Jacobian (needed for acceleration mapping)
        // J_dot appears in: ẍ = J(q)q̈ + J̇(q,q̇)q̇
        Eigen::MatrixXd J_dot_full = pin::getFrameJacobianTimeVariation(
            model_, data_, ee_frame_id_, pin::LOCAL_WORLD_ALIGNED);
        
        Eigen::MatrixXd J_dot(4, 3);
        J_dot.topRows(3) = J_dot_full.topRows(3);
        J_dot.row(3) = J_dot_full.row(3);
        
        // ====================================================================
        // OPERATIONAL SPACE CONTROL - Task-space PD controller
        // ====================================================================
        
        // Current end-effector POSITION (X-Y-Z)
        Eigen::Vector3d x = data_.oMf[ee_frame_id_].translation();
        
        // Current end-effector ORIENTATION (extract roll angle from rotation matrix)
        // data_.oMf[ee_frame_id_].rotation() returns 3×3 rotation matrix
        // For roll around X-axis: extract from rotation matrix
        // Using rotation matrix to euler angles: roll = atan2(R(2,1), R(2,2))
        Eigen::Matrix3d R = data_.oMf[ee_frame_id_].rotation();
        double roll = std::atan2(R(2,1), R(2,2));  // Roll angle (rotation around X-axis)
        
        // Combine position and orientation into 4D task-space vector
        Eigen::Vector4d x_current;
        x_current << x[0], x[1], x[2], roll;
        
        // Desired task-space state (position + roll)
        Eigen::Vector4d x_des_full;
        x_des_full << x_des_[0], x_des_[1], x_des_[2], roll_des_;
        
        // Position + orientation error: how far are we from target?
        Eigen::Vector4d x_err = x_des_full - x_current;
        
        // Wrap roll error to [-π, π] range
        // Important: orientation errors should be wrapped to avoid 360° rotations
        while (x_err[3] > M_PI) x_err[3] -= 2*M_PI;
        while (x_err[3] < -M_PI) x_err[3] += 2*M_PI;
        
        // Velocity error: how fast are we moving toward target?
        // Current velocity: J*v (map joint velocities to task-space velocity)
        Eigen::Vector4d xdot_err = xdot_des_ - J * v;
        
        // PD control law in task space:
        // Desired acceleration = feedforward + P term + D term
        // This is what we WANT the end-effector to do (position + orientation)
        Eigen::Vector4d x_acc_des = xddot_des_ + Kp_ * x_err + Kd_ * xdot_err;
        
        // ====================================================================
        // INVERSE DYNAMICS - Compute required torques
        // ====================================================================
        
        // Mass matrix (inertia): B(q) such that τ = B(q)q̈ + n(q,q̇)
        // Composite Rigid Body Algorithm - O(n) complexity
        // Symmetric positive-definite matrix
        Eigen::MatrixXd B = pin::crba(model_, data_, q);
        
        // Nonlinear effects: Coriolis, centrifugal, gravity
        // n(q,v) = C(q,v)v + g(q)
        // Recursive Newton-Euler Algorithm - O(n) complexity
        // We pass zero acceleration to get only bias terms
        Eigen::VectorXd n = pin::rnea(model_, data_, q, v, Eigen::VectorXd::Zero(model_.nv));
        
        // ====================================================================
        // TASK-SPACE TO JOINT-SPACE MAPPING
        // ====================================================================
        // We have desired task-space acceleration (x_acc_des) - 4D vector [ax,ay,az,alpha_x]
        // We need joint acceleration (qddot) that achieves it - 3D vector [q̈₁,q̈₂,q̈₃]
        //
        // From kinematics: ẍ = J(q)q̈ + J̇(q,q̇)q̇
        // Rearrange: q̈ = J⁺(ẍ - J̇q̇)
        //
        // For 3-DOF robot controlling 4D task (position + orientation):
        //   - J is 4×3 (4 task DOF, 3 joint DOF)
        //   - System is UNDERDETERMINED (more tasks than joints - impossible!)
        //   
        //   Wait... 4 tasks with 3 joints? This seems impossible!
        //   
        //   Actually, it IS possible because of your special design:
        //     • Joint0 (X-rotation) PRIMARILY controls roll
        //     • Joint1+Joint2 (Z-rotations) PRIMARILY control X,Y,Z position
        //   
        //   The tasks are "weakly coupled" - different joints affect different tasks.
        //   Pseudoinverse finds best-fit solution that minimizes error across all tasks.
        //
        //   If task cannot be perfectly achieved (all 4 DOFs simultaneously):
        //     → Pseudoinverse prioritizes based on singular value decomposition
        //     → Typically position has stronger influence than orientation
        //     → Result: Good position tracking, approximate orientation tracking
        //
        // For TRUE redundancy with guaranteed task achievement, we'd need 4+ joints.
        // With 3 joints and 4 tasks, this is actually OVERCONSTRAINED.
        // But pseudoinverse handles this gracefully - finds best compromise!
        
        Eigen::Vector3d qddot_task = J.completeOrthogonalDecomposition().solve(
            x_acc_des - J_dot * v
        );
        
        // Alternative: Use damped least squares for better numerical stability
        // double damping = 0.01;
        // Eigen::Matrix3d JTJ_damped = J.transpose() * J + damping * Eigen::Matrix3d::Identity();
        // qddot_task = JTJ_damped.ldlt().solve(J.transpose() * (x_acc_des - J_dot * v));
        
        // Check condition number to monitor how well-posed the problem is
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
        double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
        if (cond > 100) {  // Condition number > 100 indicates near-singularity
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Jacobian poorly conditioned! cond(J) = %.1f", cond);
        }
        
        // Control torque using inverse dynamics:
        // τ = B(q)q̈_des + n(q,q̇)
        // This is the torque that, if applied, produces desired acceleration
        Eigen::VectorXd u = B * qddot_task + n;
        
        // ====================================================================
        // FORWARD DYNAMICS - Compute actual resulting acceleration
        // ====================================================================
        // Given applied torque u, what acceleration actually occurs?
        // From equation of motion: B(q)q̈ = τ - n(q,q̇)
        // Solve for q̈: q̈ = B⁻¹(τ - n)
        //
        // Why not just use qddot_task?
        //   - Numerical errors, model mismatches, external forces
        //   - In real robots: motor limits, friction, delays
        //   - This simulates the "true" physics
        
        Eigen::VectorXd qddot = B.ldlt().solve(u - n);  // LDLT decomposition for symmetric B
        
        // ====================================================================
        // RETURN DERIVATIVE: dstate/dt = [v, qddot]
        // ====================================================================
        // ODE integrator needs time derivative of state
        // d/dt[q, v] = [v, qddot]
        //   - Position changes at rate = velocity
        //   - Velocity changes at rate = acceleration
        
        dstate.resize(state.size());
        dstate << v, qddot;  // Concatenate into single vector
    }
    
    // ------------------------------------------------------------------------
    // INTEGRATION STEP AND PUBLISH - Main simulation loop
    // ------------------------------------------------------------------------
    // Called by timer every dt_ seconds
    // Performs one timestep of simulation and publishes result
    
    void step_and_publish()
    {
        // ====================================================================
        // PRE-INTEGRATION: CLAMP POSITIONS TO JOINT LIMITS
        // ====================================================================
        // Ensure we START integration within valid range
        // Prevents accumulated numerical errors from violating limits
        
        state_.head(model_.nq) = state_.head(model_.nq).cwiseMax(q_min_).cwiseMin(q_max_);
        // cwiseMax/Min: element-wise maximum/minimum
        // Equivalent to: q[i] = max(q_min[i], min(q[i], q_max[i]))
        
        // ====================================================================
        // NUMERICAL INTEGRATION - Advance simulation by one timestep
        // ====================================================================
        // Solves ODE: dstate/dt = robot_dynamics(state, t)
        // Method: Runge-Kutta 4/5 with adaptive stepsize
        //   - RK45: 4th order accurate, 5th order error estimate
        //   - Adaptive: automatically adjusts internal steps for accuracy
        //   - More accurate than simple Euler: state_new = state_old + dt*f(state)
        
        using namespace boost::numeric::odeint;
        
        // Create stepper object (stateless, can be reused)
        runge_kutta_dopri5<state_type> stepper;
        
        // Integrate from time_ to time_+dt_
        // integrate_adaptive automatically chooses internal substeps
        // Absolute tolerance: 1e-6, Relative tolerance: 1e-6
        integrate_adaptive(
            make_controlled(1e-6, 1e-6, stepper),  // Error tolerances
            [this](const state_type &x, state_type &dxdt, double t) {
                this->robot_dynamics(x, dxdt, t);  // Lambda wraps member function
            },
            state_,        // Initial state (modified in-place)
            time_,         // Start time
            time_ + dt_,   // End time
            0.001          // Max internal step size (same as Python version)
        );
        
        // ====================================================================
        // POST-INTEGRATION: ENFORCE JOINT LIMITS AGAIN
        // ====================================================================
        // Integration might slightly overshoot limits due to numerical error
        // Clamp again to ensure we END within valid range
        
        state_.head(model_.nq) = state_.head(model_.nq).cwiseMax(q_min_).cwiseMin(q_max_);
        
        // ====================================================================
        // VELOCITY LIMITING AT BOUNDARIES
        // ====================================================================
        // If joint is at limit AND moving further into limit, stop it
        // Simulates hard stops (mechanical joint limits)
        
        for (int i = 0; i < model_.nq; ++i) {
            double q = state_[i];              // Joint position
            double v = state_[model_.nq + i];  // Joint velocity
            
            // At lower limit and moving downward? Stop.
            if (q <= q_min_[i] && v < 0.0) {
                state_[model_.nq + i] = 0.0;
            }
            // At upper limit and moving upward? Stop.
            else if (q >= q_max_[i] && v > 0.0) {
                state_[model_.nq + i] = 0.0;
            }
        }
        
        // Increment simulation time
        time_ += dt_;
        
        // ====================================================================
        // EXTRACT STATE FOR PUBLISHING
        // ====================================================================
        Eigen::VectorXd q = state_.head(model_.nq);
        Eigen::VectorXd v = state_.tail(model_.nv);
        
        // ====================================================================
        // BUILD AND PUBLISH JOINT_STATE MESSAGE
        // ====================================================================
        // This message is the primary output of the simulator
        // Consumed by robot_state_publisher (TF transforms) and RViz (visualization)
        
        auto msg = sensor_msgs::msg::JointState();
        
        // Header: timestamp and coordinate frame
        msg.header.stamp = this->now();  // Current ROS time
        msg.header.frame_id = "";        // Not used for joint_states (TF handles frames)
        
        // Joint names (must match URDF)
        msg.name = joint_names_;
        
        // Convert Eigen vectors to std::vector for ROS message
        msg.position.assign(q.data(), q.data() + q.size());
        msg.velocity.assign(v.data(), v.data() + v.size());
        msg.effort.clear();  // No torque data in this simulation
        
        // Broadcast message on /joint_states topic
        joint_pub_->publish(msg);
        
        // ====================================================================
        // LOGGING - Print status every 1 second
        // ====================================================================
        // Avoids flooding console with messages (would print 100 times/sec otherwise)
        
        if (static_cast<int>(time_ * 100) % 100 == 0) {  // Every 100 timesteps
            // Recompute kinematics for logging (not used elsewhere)
            pin::forwardKinematics(model_, data_, q, v);
            pin::updateFramePlacements(model_, data_);
            
            // Current end-effector position (full 3D)
            Eigen::Vector3d x = data_.oMf[ee_frame_id_].translation();
            
            // Current roll angle
            Eigen::Matrix3d R = data_.oMf[ee_frame_id_].rotation();
            double roll = std::atan2(R(2,1), R(2,2));
            
            // Position error magnitude
            double pos_err = (x_des_ - x).norm();
            
            // Orientation error magnitude
            double roll_err = std::abs(roll_des_ - roll);
            // Wrap to [-π, π]
            while (roll_err > M_PI) roll_err -= 2*M_PI;
            while (roll_err < -M_PI) roll_err += 2*M_PI;
            roll_err = std::abs(roll_err);
            
            // Convert joint angles to degrees for readability
            double q1_deg = q[0] * 180.0 / M_PI;
            double q2_deg = q[1] * 180.0 / M_PI;
            double q3_deg = q[2] * 180.0 / M_PI;
            
            // Check if any joint is at its limit
            std::vector<std::string> limits;
            if (std::abs(q[0] - q_min_[0]) < 0.02) limits.push_back("J1=MIN");
            if (std::abs(q[0] - q_max_[0]) < 0.02) limits.push_back("J1=MAX");
            if (std::abs(q[1] - q_min_[1]) < 0.02) limits.push_back("J2=MIN");
            if (std::abs(q[1] - q_max_[1]) < 0.02) limits.push_back("J2=MAX");
            if (std::abs(q[2] - q_min_[2]) < 0.02) limits.push_back("J3=MIN");
            if (std::abs(q[2] - q_max_[2]) < 0.02) limits.push_back("J3=MAX");
            
            // Build limit warning string
            std::string limit_str;
            if (!limits.empty()) {
                limit_str = " [";
                for (size_t i = 0; i < limits.size(); ++i) {
                    limit_str += limits[i];
                    if (i < limits.size() - 1) limit_str += ", ";
                }
                limit_str += "]";
            }
            
            RCLCPP_INFO(this->get_logger(),
                "t=%5.2fs | q=[%+6.1f°, %+6.1f°, %+6.1f°] | EE=[%+.3f, %+.3f, %+.3f] roll=%+6.1f° | pos_err=%.4f roll_err=%.2f°%s",
                time_, q1_deg, q2_deg, q3_deg, x[0], x[1], x[2], roll*180.0/M_PI, 
                pos_err, roll_err*180.0/M_PI, limit_str.c_str());
        }
    }
    
    // ------------------------------------------------------------------------
    // MEMBER VARIABLES - Store node state
    // ------------------------------------------------------------------------
    
    // Pinocchio model and data
    pin::Model model_;           // Robot kinematic/dynamic model
    pin::Data data_;             // Workspace for computations
    pin::FrameIndex ee_frame_id_; // Numerical ID of end-effector frame
    
    // Joint information
    std::vector<std::string> joint_names_;  // Names from URDF
    
    // Control parameters
    Eigen::Vector3d x_des_;      // Target end-effector position (m) [X, Y, Z]
    double roll_des_;            // Target roll angle around X-axis (rad)
    Eigen::Vector4d xdot_des_;   // Target velocity [vx, vy, vz, omega_x] (m/s, rad/s)
    Eigen::Vector4d xddot_des_;  // Target acceleration [ax, ay, az, alpha_x] (m/s², rad/s²)
    Eigen::Matrix4d Kp_;         // Position + orientation gain matrix (4×4)
    Eigen::Matrix4d Kd_;         // Velocity + angular velocity damping matrix (4×4)
    
    // Joint limits
    Eigen::Vector3d q_min_;      // Lower joint limits (rad)
    Eigen::Vector3d q_max_;      // Upper joint limits (rad)
    
    // Simulation state
    state_type state_;           // Current [q, v] vector
    double time_;                // Current simulation time (s)
    double dt_;                  // Timestep (s)
    
    // ROS2 components
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;  // Publisher
    rclcpp::TimerBase::SharedPtr timer_;  // Timer for periodic execution
};

// ============================================================================
// MAIN FUNCTION - Entry point
// ============================================================================
int main(int argc, char** argv)
{
    // Initialize ROS2 communication
    // Must be called before any ROS2 functions
    // Connects to ROS daemon, sets up signal handlers
    rclcpp::init(argc, argv);
    
    // Create node instance (calls constructor)
    auto node = std::make_shared<PendulumSimulator>();
    
    // Spin: enter event loop (BLOCKS HERE until Ctrl+C)
    // Executor checks timers/callbacks and invokes them as needed
    // This is what keeps your program running and responding to events
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Exception in spin: %s", e.what());
    }
    
    // Cleanup: shutdown ROS2 (disconnects from daemon)
    rclcpp::shutdown();
    
    return 0;
}

// ============================================================================
// COMPILATION AND EXECUTION
// ============================================================================
// 
// CMakeLists.txt dependencies:
//   - rclcpp
//   - sensor_msgs
//   - pinocchio
//   - Eigen3
//   - Boost (system, odeint)
//
// Build:
//   $ cd ~/ros2_workspace
//   $ colcon build --packages-select your_package_name
//
// Run:
//   $ source install/setup.bash
//   $ ros2 run your_package_name pendulum_simulator
//
// Visualize:
//   $ ros2 launch your_package_name rviz.launch.py
//
// Check topics:
//   $ ros2 topic list
//   $ ros2 topic echo /joint_states
//
// ============================================================================
// EXECUTION FLOW DIAGRAM
// ============================================================================
//
// main()
//   ↓
// rclcpp::init()  ← ROS2 initializes
//   ↓
// node = PendulumSimulator()  ← Constructor executes
//   ├─ Load URDF
//   ├─ Setup Pinocchio model
//   ├─ Create publisher (/joint_states)
//   └─ Start timer (calls step_and_publish every 0.01s)
//   ↓
// rclcpp::spin(node)  ← **BLOCKS HERE** - Event loop runs
//   │
//   │ Timer fires every 10ms in background:
//   │   └─> step_and_publish()
//   │         ├─ Clamp positions to limits
//   │         ├─ Integrate dynamics (Boost OdeInt)
//   │         ├─ Clamp again after integration
//   │         ├─ Enforce velocity limits at boundaries
//   │         ├─ Build JointState message
//   │         ├─ Publish to /joint_states
//   │         └─ Log status (every 1 second)
//   │
//   ↓ (User presses Ctrl+C)
// rclcpp::shutdown()  ← Cleanup and exit
//
// ============================================================================
// ROS2 MESSAGE FLOW
// ============================================================================
//
//     ┌─────────────────────────────┐
//     │   PendulumSimulator Node    │
//     │  (this program)             │
//     └──────────┬──────────────────┘
//                │ publishes
//                ↓
//         /joint_states topic
//    (sensor_msgs/msg/JointState)
//                │
//                ├──→ robot_state_publisher
//                │    (converts joint states to TF transforms)
//                │    
//                ├──→ RViz2
//                │    (visualizes robot)
//                │
//                └──→ Your custom nodes
//                     (can subscribe to joint data)
//
// ============================================================================
// KEY DIFFERENCES FROM PYTHON VERSION
// ============================================================================
//
// 1. **Memory Management**:
//    - Python: Automatic garbage collection
//    - C++: Manual with smart pointers (shared_ptr, make_shared)
//
// 2. **Type System**:
//    - Python: Dynamic typing (variables can change type)
//    - C++: Static typing (must declare types explicitly)
//
// 3. **Arrays/Vectors**:
//    - Python: Lists, NumPy arrays
//    - C++: std::vector, Eigen::VectorXd, Eigen::MatrixXd
//
// 4. **Integration**:
//    - Python: scipy.integrate.solve_ivp
//    - C++: Boost.OdeInt (boost::numeric::odeint)
//
// 5. **Lambda Functions**:
//    - Python: lambda x: f(x)
//    - C++: [this](const state_type &x, state_type &dxdt, double t) { ... }
//
// 6. **Error Handling**:
//    - Python: try/except
//    - C++: try/catch (less common in this code)
//
// 7. **String Formatting**:
//    - Python: f"Value: {x:.3f}"
//    - C++: "Value: %.3f", x (printf-style)
//
// ============================================================================
// PERFORMANCE NOTES
// ============================================================================
//
// C++ version is typically 5-10x faster than Python because:
//   - No interpreter overhead
//   - Better compiler optimizations
//   - Eigen library uses SIMD instructions (AVX, SSE)
//   - No GIL (Global Interpreter Lock) issues
//   - Stack allocation instead of heap when possible
//
// For real-time robotics:
//   - C++ is preferred for control loops >100 Hz
//   - Python works well for higher-level planning/monitoring
//   - Both can coexist in same ROS2 system
//
// ============================================================================
// DEBUGGING TIPS
// ============================================================================
//
// 1. Check joint_states are publishing:
//    $ ros2 topic hz /joint_states
//    average rate: 100.000
//
// 2. Inspect message content:
//    $ ros2 topic echo /joint_states
//
// 3. Verify node is running:
//    $ ros2 node list
//    /pendulum_simulator
//
// 4. Check parameter values:
//    $ ros2 param list /pendulum_simulator
//    $ ros2 param get /pendulum_simulator urdf_path
//
// 5. View computation graph:
//    $ rqt_graph
//    (Shows nodes, topics, connections)
//
// 6. Common issues:
//    - URDF not found → Check path in launch file or parameters
//    - No visualization → Ensure robot_state_publisher is running
//    - Unstable motion → Tune Kp/Kd gains, check joint limits
//    - Spinning joints → Verify q_min/q_max are enforced
//
// ============================================================================
// EXTENDING THIS CODE
// ============================================================================
//
// Add trajectory tracking:
//   - Replace static x_des with time-varying trajectory
//   - Update x_des_, xdot_des_, xddot_des_ in step_and_publish()
//
// Add obstacle avoidance:
//   - Compute null-space projection: N = I - J⁺J
//   - Add secondary task: qddot = qddot_task + N * qddot_secondary
//
// Add joint torque limits:
//   - Clamp u before forward dynamics
//   - Simulate actuator saturation
//
// Add external forces:
//   - Modify rnea call to include external wrenches
//   - Simulate contact, wind, etc.
//
// Add sensor noise:
//   - Perturb q and v before publishing
//   - Simulate realistic sensor inaccuracies
//
// Switch to velocity control:
//   - Instead of torque τ, command joint velocities
//   - Use different integration scheme
//
// ============================================================================