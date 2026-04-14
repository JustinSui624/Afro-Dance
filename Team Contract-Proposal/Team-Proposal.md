# AfroDance Learn — Senior Design Project Proposal

> Converted from the submitted PDF proposal for version-control friendliness.

## Cover Sheet

- **Project Title:** AfroDance Learn: An Interactive AI-Based Afro Dance Learning Platform  
- **Team Name:** Digital Dynamos  
- **Team Members and Roles:**
  - Luis Borruel — Front-End Developer  
  - Justin Sui — Back-End Developer  
  - Mason Bush — Quality Assurance & Testing  
- **Faculty Advisor:** Professor Rujeko Dumbutshena  
- **Date of Resubmission:** February 11, 2026  

## Abstract

AfroDance Learn is a human motion analysis software system designed to support the learning and practice of African dance styles through real-time visual feedback and movement evaluation. Many individuals interested in learning Afro dance lack access to professional instructors, structured environments, or affordable instructional tools. Existing dance applications emphasize entertainment rather than education or cultural understanding.

AfroDance Learn addresses this gap by using computer vision techniques to track a user’s body movements via a webcam or Xbox 360/One Kinect device and compare them against a professional instructor’s performance. The system provides step-by-step instructions, instructor overlay visualization, and accuracy scoring based on body posture and joint movement. The project aims to promote accessible dance education while demonstrating applied computer science principles and professional software development practices.

## Table of Contents

1. Team Contract  
2. Introduction and Motivation  
3. Literature Survey  
4. Proposed Work  
5. Product Backlog  
6. Project Plan  
7. GitHub Repository  
8. References  
9. Signatures  

---

## 1. Team Contract

The team contract has been completed and signed by all members. The contract defines team objectives, individual roles and responsibilities, communication guidelines, meeting schedules, configuration management protocols, and conflict resolution procedures.

> See: `team-contract.md`

---

## 2. Introduction and Motivation

### Problem Formation

African dance is deeply rooted in cultural history and expression, yet many learners lack access to professional instruction, practice spaces, or structured feedback. Geographic limitations, financial barriers, and scheduling conflicts often prevent individuals from participating in in-person classes. Existing digital dance applications primarily focus on entertainment and scoring rather than instruction, accuracy, or cultural education.

### Rationale

Recent advancements in computer vision and human pose estimation enable analysis of body movement using consumer hardware such as webcams and depth-sensing devices like the Xbox Kinect. These technologies make it possible to deliver educational, interactive dance instruction at home while maintaining accuracy and accessibility.

### Proposed Solution

AfroDance Learn is a human motion analysis system that captures user movement through a webcam or Xbox 360/One Kinect device. The software compares user movements to a professional African dance instructor’s reference performance. Users receive real-time visual overlays, step-by-step guidance, and scoring based on joint angles, posture, and timing.

### Importance and Impact

This project promotes cultural preservation, accessibility, and inclusive learning while providing a practical application of computer science concepts such as computer vision, software architecture, and quality assurance. AfroDance Learn offers an affordable educational tool while serving as a strong demonstration of collaborative software development.

---

## 3. Literature Survey

Human pose estimation has become a key area of research in computer vision. Google’s MediaPipe Pose framework enables real-time pose tracking using standard RGB cameras and has been successfully applied in fitness and motion analysis applications. This technology forms the foundation for webcam-based tracking in AfroDance Learn.

Research on motion-based learning systems shows that real-time visual feedback significantly improves motor skill acquisition. Dance training systems that compare learner movements to expert references help users self-correct posture and timing more effectively than passive video instruction.

Commercial dance applications such as *Just Dance* demonstrate the motivational value of interactive movement-based software; however, these systems prioritize entertainment and competition over instructional feedback or cultural education. AfroDance Learn differentiates itself by focusing on learning accuracy, progressive instruction, and cultural context.

---

## 4. Proposed Work

### Plan of Work

The project will be developed as a Python-based desktop application that performs real-time human motion analysis. The system will process live video input, extract body pose landmarks, and compare them against instructor reference data.

### Technologies and Languages

- **Programming Language:** Python  
- **Motion Sensors:** Webcam and Xbox 360/One Kinect  
- **Version Control:** GitHub  

### Features

- Real-time pose tracking  
- Step-by-step dance instruction  
- Full dance sequence mode  
- Accuracy scoring based on joint-angle comparison  
- Adaptive, resizable user interface  

### Target Audience

- Beginner and intermediate African dance learners  
- Students and educators interested in cultural dance instruction  
- Users seeking at-home movement learning tools  

---

## 5. Product Backlog

| Priority | Feature | Description | Responsible |
|---|---|---|---|
| High | Pose Tracking | Capture and track user movement | Justin |
| High | Instructor Reference Processing | Convert instructor video into reference data | Justin |
| Medium | UI Resizing & Layout | Responsive window and display | Luis |
| Medium | Step-by-Step Mode | Gather and segment dances into instructional steps | Mason |
| Low | Cultural Education Content | African dance history integration | Mason |

---

## 6. Project Plan

| Milestone | Description | Target Date |
|---|---|---|
| Proposal Approval | Advisor approval obtained | Week 1 |
| System Setup | Development environment ready | Week 2 |
| Reference Extraction | Instructor motion processing | Week 3 |
| Core Tracking | Live pose detection operational | Week 4 |
| Overlay & Scoring | Visual feedback implemented | Week 5 |
| Step & Sequence Modes | Instructional logic complete | Week 6 |
| QA & Testing | Functional and usability testing | Week 7 |
| Final Demo | Presentation and documentation | Week 8 |

---

## 7. GitHub Repository

- Repository: `Afro-Dance` (as listed in the proposal)

---

## 8. References

- Chan, J. C. P., et al. *A Virtual Reality Dance Training System Using Motion Capture Technology.* IEEE Transactions on Learning Technologies.  
- Google. *MediaPipe Pose: Real-Time Human Pose Estimation.*  
- Anderson, E. F., et al. *Interactive Dance Systems for Learning and Performance.* IEEE Computer Graphics and Applications.  

---

## 9. Signatures

| Team Member | Date |
|---|---|
| Luis Borruel | 2/1/2026 |
| Justin Sui | 2/1/2026 |
| Mason Bush | 2/1/2026 |
| Prof Rujeko Dumbutshena | 2/5/2026 |
