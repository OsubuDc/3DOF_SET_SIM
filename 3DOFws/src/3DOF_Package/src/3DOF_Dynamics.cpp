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
#include <boost/numeric/odeint.hpp>       // ODE solver

//using is used to avoid writing std::chrono::duration every time. You can write 10ms instead of std::chrono::duration<int, std::milli>(10)
using namespace std::chrono_literals;     // Enables time literals like 10ms, 1s
                                          // std = standard library
                                          // chrono = time-related functions and types

// Pinocchio namespace shortcuts
namespace pin = pinocchio;                // pin::Model instead of pinocchio::Model

// State vector type: [q1, q2, v1, v2] 
typedef Eigen::VectorXd state_type; //typedef is used to create an alias for a data type. Here, state is an alias for Eigen::VectorXd, which is a dynamic-sized vector from the Eigen library. This alias simplifies the code and improves readability by allowing you to use state_type instead of repeatedly writing Eigen::VectorXd throughout the code.
                                    //A "type" is a classification identifying one of various types of data, such as integer, floating-point, or user-defined types like classes and structures. In programming, types define the operations that can be performed on the data and how the data is stored in memory.
                                    //Difference between a class and a type: A class is a blueprint for creating objects, encapsulating data and behavior, while a type is a broader concept that defines the nature of data, including primitive types (like int, float) and user-defined types (like classes and structs).

Class 3DOF_Dynamics : public rclcpp::Node //inherits from rclcpp::Node, gaining all ROS2 node capabilities. We use public inheritance so that all public and protected members of rclcpp::Node are accessible in 3DOF_Dynamics.
{
    
    private: //visibility specifier: members declared under private can only be accessed from within the class itself.
        
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
        rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;  // Publisher publishes message of type <something>. SharedPtr is a smart pointer that retains shared ownership of an object through a pointer. 
                                                                                // Several SharedPtr objects may own the same object. The object is destroyed and its memory deallocated when either of the following happens: 
                                                                                // the last remaining SharedPtr owning the object is destroyed; or the last remaining SharedPtr owning the object is assigned another pointer via operator= or reset().
                                                                                // In this case, joint_pub_ is a shared pointer to a ROS2 publisher that publishes messages of type sensor_msgs::msg::JointState.
                                                                                // JointState message contains information about the state of joints, including their names, positions, velocities, and efforts.
                                                                                // When system stops using this publisher, the memory allocated for it will be automatically released because of SharedPtr. This way you don't have to manually manage memory.
        rclcpp::TimerBase::SharedPtr timer_;  // Timer for periodic execution

        void robot_dynamics(const state_type &state, state_type &dstate, double t) //const means the function does not modify the input state.  double t is the current time.
        {
            Eigen::VectorXd q = state.head(model_.nq);  // First nq elements are positions
            Eigen::VectorXd v = state.tail(model_.nv);  // Last nv elements are velocities

            pin::forwardKinematics(model_, data_, q, v); // Update kinematics with current q and v. velocity is passed to calculate the jacobian time derivative.
            pin::updateFramePlacements(model_, data_); // Update frame placements (end-effector position)
            // in data_ you have data_.oMf which contains the transformation matrices of all frames in the model. You also have data_.oMi which contains the transformation matrices of all joints in the model.
            // The difference between a frame and a joint is that a joint is a connection between two links that allows relative motion, while a frame is a reference point or coordinate system attached to a link or joint for defining positions and orientations in space.
            // A frame can be associated with a joint, but it can also be associated with other parts of the robot, such as the end-effector or sensors.
            // forwardKinematics computes the spatial kinematics of all joints. Updates data_.oMi, data_.v (if v is provided), data_.a (if a is provided), etc.
            // v is rotational speed (for revolute joints) or linear speed (for prismatic joints) of each joint. In data_.v you have the spatial velocities of all joints (spatial means in x,y,z and rotational velocities around x,y,z)
            // data_.a is calculated with this formula: ai​=Xiparent​aparent​+Si​q¨​i​+vi​×(Si​q˙​i​) (recursive spatial acceleration formula)
            // updateFramePlacements updates the the pose of all frames, it used data_.oMi to compute data_.oMf.
            
            //computeFrameJacobian(here you write the model, data, joint positions, frame id, reference frame)
            Eigen::MatrixXd J_full = pin::computeFrameJacobian(model_,data_,q,ee_frame_id_,pin::LOCAL_WORLD_ALIGNED); // Compute full 6xN Jacobian for end-effector frame in world-aligned coordinates
            //end-effector position
            Eigen::Vector3d x = data.oMf[ee_frame_id_].translation (); // data.oMf[i] gives the transformation matrix of frame i. .translation() extracts the translation vector (position) from the transformation matrix.
            
            double z_threshold = 0.005; // 5mm threshold for "near X-Y plane"
            bool near_xy_plane = std::abs(x[2]) < z_threshold; //check if z coordinate of end-effector is less than threshold
                                                               // If z < treshold, bool will be true, else false.
            Eigen::VectorXd x_acc_des; //desired end-effector acceleration
            Eigen::MatrixXd J; //Jacobian matrix. Xd means the size of the matrix is dynamic (can change at runtime)
            Eigen::MatrixXd J_dot; //time derivative of Jacobian matrix

            if (near_xy_plane)
            {
                
            }
        }

        
    public: //visibility specifier: members declared under public can be accessed from outside the class.
        //create constructor: Automatically called when an object of the class is created.
        3DOF_Dynamics() // after : you have initializer list. This is used to initialize member variables before the constructor body executes.
                        // A member is a variable or function that is part of a class.
            : rclcpp::Node("3DOF_Dynamics_Node"), //calls constructor of parent class rclcpp::Node with a node name
                time_(0.0),                  // Start simulation time at zero. This is a member variable initialization.
                dt_(0.01)                    // Timestep: 100 Hz. This is a member variable initialization.
        {
            this->declare_parameter<std::string>("urdf_path", 
            "/3DOF_SET_SIM/3DOFws/urdf/AssemblyVersion3.2.urdf");
            this->declare_parameter<std::string>("mesh_dir",      
            "/3DOF_SET_SIM/3DOFws/mesh");
            //you need to use declare_parameter to register parameters with the ROS2 parameter server. This allows you to define parameters that can be set externally (e.g., via command line or configuration files) and retrieved within your node.
            //(here you write the parameter name as it will appear in ROS2 , and here you write the default value of the parameter)
            //Parameters are used to configure the behavior of nodes without changing the code. It is a ros2 feature, not a c++ feature.        
            // you write string because the parameter value is a string (file path).

            // Retrieve parameter values into local variables
            std::string urdf_path = this->get_parameter("urdf_path").as_string(); // get_parameter retrieves the value of a parameter from the ROS2 parameter server.
            std::string mesh_dir = this->get_parameter("mesh_dir").as_string(); // you need to do .as_string() for when the parameter value is overriden externally (e.g., via command line or config files) to convert it to the appropriate type.
                                                                                // std::string urdf_path is a local variable that stores the value of the "urdf_path" parameter, the type is a string.
            RCLCPP_INFO(this->get_logger(), "Loading URDF from: %s", urdf_path.c_str()); // RCLCPP_INFO is a macro for logging informational messages in ROS2.
                                                                                          // this->get_logger() retrieves the logger associated with the current node.
                                                                                          // "Loading URDF from: %s" is the format string, where %s will be replaced by the value of urdf_path.
                                                                                          // urdf_path.c_str() converts the std::string urdf_path to a C-style string (const char*) for compatibility with the logging function.
                                                                                          // A macro is a preprocessor directive that defines a piece of code that can be reused multiple times throughout the program. In this case, RCLCPP_INFO is a macro that simplifies the process of logging informational messages in ROS2.
        
            pin::urdf::buildModel(urdf_path, model_); //parse URDF file and build kinematic tree
            data_ = pin::Data(model_);; // Create data structure (allocates memory for intermediate calculations)
                                        // _ is used to indicate that the variable is a member variable of the class. It helps to distinguish member variables from local variables or function parameters with the same name.

            model_.gravity.linear(Eigen::Vector3d(0.0, -9.81, 0.0)); // Set gravity vector in world frame (Y-up convention)

            for (size_t i =1; i<model_.names.size(); i++){ // for(initialization; condition; increment) size_t is an unsigned integer type used for array indexing and loop counting. It is guaranteed to be able to represent the size of any object in bytes.
                //loop body
                joint_names.push_back(model_.names[i]); //push back adds an element to the end of the vector joint_names_
            }
            RCLCPP_INFO(this->get_logger(), "Found %zu joints in URDF.", joint_names.size()); // %zu is used to print size_t type variable
            for (const auto& name : joint_names_) { //const means cannot be modified, auto means the compiler will automatically deduce the type of the variable name based on the type of elements in joint_names_ vector, & means name is a reference to the actual element in the vector (avoids copying)
                RCLCPP_INFO(this->get_logger()," - %s", name.c_str()); // %s is used to print C-style string. It takes the first argument after the format string and replaces %s with that value.
            }

            std::string ee_frame_name = "EndEffector"

            if(model_.existFrame(ee_frame_name)){ //existFrame checks if a frame with the given name exists in the model
                ee_frame_id = model_.getFrameId(ee_frame_name); //getFrameId returns the numerical index of the frame with the given name
                RCLCPP_INFO(this->get_logger(),"Found end-effector frame: %s (ID: %ld)", // %ld is used to print long integer type variable, takes the second argument after the format string and replaces %ld with that value.
                            ee_frame_name.c_str(), ee_frame_id_);
            } else {
                RCLCPP_ERROR(this->get_logger(),"Could not find frame '%s'", ee_frame_name.c_str()); //c_string is used to convert std::string to C-style string because RCLCPP_ERROR expects a C-style string as input.
                RCLCPP_INFO(this->get_logger(),"Available frames:");
                for (size_t i =0; i<model_.frames.size(); i++){
                    RCLCPP_INFO(this->get_logger(),"  %zu: %s", i, model_.frames[i].name.c_str()); //gets frame i and retrieves its name
                }
            ee_frame_id = model_.frames.size() -1; //set to last frame as fallback
            RCLCPP_WARN(this->get_logger(),"Using fallback frame: %s",
                        model_.frames[ee_frame_id_].name.c_str());
            }

            pos_desired = Eigen::Vector3d(0.2,0.2,0.2); //desired end-effector position
            vel_desired = Eigen::Vector3d(0.0,0.0,0.0); //desired end-effector velocity
            acc_desired = Eigen::Vector3d(0.0,0.0,0.0); //desired end-effector acceleration
            roll_des_ = 0.0; //desired roll angle around X-axis. Only used when Z is zero (one control parameter is lost in a 3DOF arm, meaning we can control rotation around first joint when the end-effector is in the x-y plane)

            Kp_ = Eigen::Matrix3d::Identity() * 50.0;  // 3×3 for X, Y, Z
            Kd_ = Eigen::Matrix3d::Identity() * 40.0;

            Kp_roll_ = 30.0; // Roll position gain
            Kd_roll_ = 25.0; // Roll velocity damping

            q_min_ = Eigen::Vector3d(-M_PI, -1.57, -0.1);   //joint lower limits
            q_max_ = Eigen::Vector3d(M_PI, 1.57, -2.6);     //joint upper limits
                                                            //M_PI is a constant defined in cmath representing the value of pi (3.14159...)
            RCLCPP_INFO(this->get_logger(), "Target position: [%.3f, %.3f, %.3f]", 
                        x_des_[0], x_des_[1], x_des_[2]); //%.3f is used to print floating-point numbers with 3 decimal places. The % means that the value will be replaced by the corresponding argument after the format string.
            if (std::abs(x_des_[2]) < 0.005) {
                RCLCPP_INFO(this->get_logger(), "Target is near Z=0, will control roll angle: %.1f°", //%.1f is used to print floating-point numbers with 1 decimal place.
                            roll_des_ * 180.0/M_PI);
            }
            RCLCPP_WARN(this->get_logger(), "Joint limits: [%.0f°, %.0f°, %.0f°] to [%.0f°, %.0f°, %.0f°]", //%.0f is used to print floating-point numbers with no decimal places.
                        q_min_[0] * 180.0/M_PI, q_min_[1] * 180.0/M_PI, q_min_[2] * 180.0/M_PI,
                        q_max_[0] * 180.0/M_PI, q_max_[1] * 180.0/M_PI, q_max_[2] * 180.0/M_PI);
        
            Eigen::Vector3d q_init(0.01, 0.01, 0.1);  // Small non-zero initial angles
            Eigen::Vector3d v_init = Eigen::Vector3d::Zero();  // Start at rest

            state_.resize(model_.nq + model_.nv);  // nq = num positions, nv = num velocities
            state_ << q_init, Eigen::Vector3d::Zero();  // Combine into single vector. << is the Eigen library's syntax for concatenating vectors. 
            
            RCLCPP_INFO(this->get_logger(), "Initial configuration: [%.1f°, %.1f°, %.1f°]",
                        q_init[0] * 180.0/M_PI, q_init[1] * 180.0/M_PI, q_init[2] * 180.0/M_PI);
            
            joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>( //create_publisher is a member function of the rclcpp::Node class that creates a publisher object for a specific message type and topic.
                "joint_states", 10);                                           //JointState is the message type, "joint_states" is the topic name, and 10 is the queue size (number of messages to buffer if subscriber is slow).
            timer_ = this->create_wall_timer( //create_wall_timer is a member function of the rclcpp::Node class that creates a timer object that triggers a callback function at a specified interval. Takes two arguments: duration and callback function.
                std::chrono::duration<double>(dt_), //duration specifies the time interval for the timer. Here, it is set to dt_ seconds (0.01s).
                std::bind(&3DOF_Dynamics::update, this) //std::bind creates a callable object that binds the member function update of the 3DOF_Dynamics class to the current instance (this).
            );                                          //When the timer expires, it will call the update function of this instance.
                                                        // a member function is a function that is defined within a class and operates on instances of that class.
                                                        // & pointer to 3DOF_Dynamics::update means we are passing the address of the update function to std::bind.
                                                        //std::bind is used to create a callable object that can be invoked later, in this case by the timer when it expires.
                                                        // this is a pointer to the current instance of the class. You need to write it because a member function pointer cannot be called by itself, you need to provide an instance of the class to call it on.
            RCLCPP_INFO(this->get_logger(), "3DOF Dynamics simulator started! Publishing at %.0f Hz", 1.0/dt_);
        };

}                                                 


    