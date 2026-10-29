# JIT-Accelerated 3D Black Hole Ray-Tracer

A real-time, interactive 3D ray-tracing simulation of a Schwarzschild black hole built in Python. Powered by **Numba JIT compilation** and **Pygame**, it calculates relativistic light bending and gravitational lensing in parallel across CPU cores to render at a smooth 60 FPS in 640x480 resolution.

---

## Features
- **General Relativity Simulation:** Iteratively traces light ray geodesics through curved spacetime around a central gravitational singularity.
- **Real-Time Mouse Orbiting:** Click and drag to smoothly orbit the camera around the black hole while keeping it mathematically locked to the center.
- **Dynamic Zoom:** Use the mouse scroll wheel to seamlessly zoom in and out of the event horizon.
- **Accretion Disk & Floor Grid:** Features a spinning, Keplerian-velocity accretion disk and a glowing orange spacetime floor grid that warps dramatically over the event horizon.
- **High-Performance JIT Kernel:** Utilizes Numba (`@njit(parallel=True)`) to execute heavy pixel-by-pixel ray tracing in parallel, bypassing Python's standard execution bottlenecks.

---

## Tech Stack
- **Language:** Python 3
- **Display & Input:** Pygame
- **Matrix Operations:** NumPy
- **Performance Acceleration:** Numba (Parallel JIT Compiler)

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kia21-ya/blackholebyme1.git](https://github.com/kia21-ya/blackholebyme1.git)
   cd blackholebyme1

    Create and activate a virtual environment:
    Bash

    python3 -m venv .venv
    source .venv/bin/activate

    Install dependencies:
    Bash

    pip install pygame numpy numba

Running the Simulation

To launch the simulator with optimal window rendering (especially on Linux/Wayland environments):
Bash

SDL_VIDEODRIVER=x11 python simulator.py

Controls

    Orbit Camera: Click and drag the left mouse button.

    Zoom In/Out: Scroll the mouse wheel up or down.

    Exit: Close the simulation window.

Project Structure

    simulator.py — Contains the core window initialization, mouse event loops, Numba parallel physics kernel, and rendering pipeline.

Demo
