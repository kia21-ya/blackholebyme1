#FOR THIS PROJECT i AM i USING AI FOR THE GENERATION OF THE THE CODE BUT THE COMMENT STATE MENTS WRITTEN ARE WRITTEN BY ME 
#SO PLEASE FEEL FREE TO CORRECT me


#ALL THE IMPORTS 
import pygame #display  buffer
import numpy as np
from numba import njit, prange
import sys
import math

#INITIALIZING OF PYGAME
pygame.init() 

#SETTING UP THE VARAIBLES

WIDTH, HEIGHT = 640, 480 #RESOLUTION OF THE SIMUTOR SCREEN
screen = pygame.display.set_mode((WIDTH, HEIGHT)) #INTIALIZING OF THE PYGAME WINDOWS
pygame.display.set_caption("JIT accelerated simulator")
pygame.event.pump() # NEW: Tells Pop!_OS the window is alive so OBS doesn't freeze it
clock = pygame.time.Clock() # Framerate clock monitor


# DEFINING THE PHYSICS CONSTANTS NEEDED FOR THE WHOLE CALCULATION
STEPS = 100 #MAXIMUM ITERATION STEPSPER LIGHT RAY
STEP_SIZE = 0.4#SPATIAL STEP DISTANCE PER ITERATION
BH_MASS = 1.5 #MASS PARAMETER OF THE CENTRAL SINGULARITY
HORIZAN_R = 2.0#EVENT HORIZAN RADIUS 
DISK_IN = 4.0#ACCELERATION DISK INNER BOUNDARY
DISK_OUT = 9.0#ACERLERATION DISK OUTER BOUNDARY

print("Block 1 of the simulation ready!!!!")

#BLOCK 2 
#SETTING OF THE CAMERA FOR THE SIMULATION VEIWING 
cam_distance = 12.0


#initial rotation angle (radian)
yaw = 0.0 #horizantal rotation 
pitch = 0.0#Vertical rotation

#mouse tracking flags
is_dragging = False
last_mouse_pos = (0, 0)


print("Block 2 of the simulation ready!!!!")

#Block 3 physic calculation machine

# --- BLOCK 3 & 4: Interactive JIT Parallel Physics Engine ---
@njit(parallel=True, fastmath=True)
def trace_rays_interactive(width, height, steps, step_size, bh_mass, horizan_r, disk_in, disk_out, cam_dist, yaw_angle, pitch_angle, sim_time, output_img):
    
    cos_y = math.cos(yaw_angle)
    sin_y = math.sin(yaw_angle)
    cos_p = math.cos(pitch_angle)
    sin_p = math.sin(pitch_angle)

    # 1. FIXED: Added the minus sign to cam_x so the camera stays locked on the dead center!
    cam_x = -cam_dist * sin_y * cos_p
    cam_y = cam_dist * sin_p
    cam_z = -cam_dist * cos_y * cos_p

    for y in prange(height):
        screen_y = 1.0 - (2.0 * y) / height
        
        for x in range(width):
            screen_x = -1.33 + (2.66 * x) / width
            
            vx = screen_x
            vy = screen_y
            vz = 1.0

            vy_rot = vy * cos_p - vz * sin_p
            vz_rot = vy * sin_p + vz * cos_p

            dx = vx * cos_y + vz_rot * sin_y
            dy = vy_rot
            dz = -vx * sin_y + vz_rot * cos_y

            norm = (dx*dx + dy*dy + dz*dz) ** 0.5
            dx /= norm; dy /= norm; dz /= norm

            px = cam_x
            py = cam_y
            pz = cam_z

            active = True

            for step in range(steps):
                if not active:
                    break

                r = (px*px + py*py + pz*pz) ** 0.5

                if r <= horizan_r:
                    output_img[y, x, 0] = 0; output_img[y, x, 1] = 0; output_img[y, x, 2] = 0
                    active = False
                    break

                cross_x = py * dz - pz * dy
                cross_y = pz * dx - px * dz
                cross_z = px * dy - py * dx
                ang_sq = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z

                accel = (3.0 * bh_mass * ang_sq) / (r ** 5)

                dx += (-px / r) * accel * step_size
                dy += (-py / r) * accel * step_size
                dz += (-pz / r) * accel * step_size

                v_norm = (dx*dx + dy*dy + dz*dz) ** 0.5
                dx /= v_norm; dy /= v_norm; dz /= v_norm

                old_py = py
                px += dx * step_size
                py += dy * step_size
                pz += dz * step_size

                # Accretion Disk Crossing Check
                if (old_py > 0.0 and py < 0.0) or (old_py < 0.0 and py > 0.0):
                    t = -old_py / dy
                    ix = px - dx * (step_size * (1.0 - t))
                    iz = pz - dz * (step_size * (1.0 - t))
                    disk_dist = (ix*ix + iz*iz) ** 0.5
                    
                    if disk_in <= disk_dist <= disk_out:
                        angle = math.atan2(iz, ix)
                        angle -= sim_time * (2.0 / math.sqrt(disk_dist))
                        pattern = math.sin(disk_dist * 8.0) * math.sin(angle * 12.0)
                        
                        if pattern > 0:
                            output_img[y, x, 0] = 255; output_img[y, x, 1] = 120; output_img[y, x, 2] = 0
                        else:
                            output_img[y, x, 0] = 150; output_img[y, x, 1] = 50; output_img[y, x, 2] = 0
                        active = False
                        break

                # 2. UPDATED: Floor Grid Crossing Check (Y = -6.0) is now Glowing Orange!
                if (old_py > -6.0 and py < -6.0) or (old_py < -6.0 and py > -6.0):
                    t = (-6.0 - old_py) / dy
                    ix = px - dx * (step_size * (1.0 - t))
                    iz = pz - dz * (step_size * (1.0 - t))
                    
                    if math.sin(ix * 1.5) > 0.92 or math.sin(iz * 1.5) > 0.92:
                        output_img[y, x, 0] = 255; output_img[y, x, 1] = 150; output_img[y, x, 2] = 0
                    else:
                        output_img[y, x, 0] = 10; output_img[y, x, 1] = 10; output_img[y, x, 2] = 10
                    active = False
                    break
            
            # 3. UPDATED: Pure Pitch-Black Space Background!
            if active:
                output_img[y, x, 0] = 0; output_img[y, x, 1] = 0; output_img[y, x, 2] = 0


def render_frame(sim_time, current_cam_dist):
    output_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    trace_rays_interactive(
        WIDTH, HEIGHT, STEPS, STEP_SIZE,
        BH_MASS, HORIZAN_R, DISK_IN, DISK_OUT,
        current_cam_dist, yaw, pitch, sim_time, output_img
    )
    return output_img

print("Block 3 & 4 of the simulation ready!!!!")


# NEW: OBS FREEZE FIX WARM-UP
print("Compiling JIT Kernel in background... (This prevents OBS from freezing!)")
dummy_img = np.zeros((1, 1, 3), dtype=np.uint8)
trace_rays_interactive(
    1, 1, 1, STEP_SIZE,
    BH_MASS, HORIZAN_R, DISK_IN, DISK_OUT,
    12.0, 0.0, 0.0, 0.0, dummy_img
)
print("JIT Compilation Complete! Starting main loop...")


# MAIN APPLICATION LOOP & MOUSE CONTROL
# --- BLOCK 5: MAIN APPLICATION LOOP & MOUSE CONTROL ---
running = True
sim_time = 0.0  # Controls the disk rotation speed

while running:
    sim_time += 0.05 
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                is_dragging = True
                last_mouse_pos = event.pos
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  
                is_dragging = False
                
        elif event.type == pygame.MOUSEMOTION:
            if is_dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]

                yaw -= dx * 0.01
                pitch += dy * 0.01

                # Lock pitch so you don't flip upside down
                pitch = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, pitch))

                last_mouse_pos = event.pos 
                
        # NEW: Handle Mouse Scroll Wheel to Zoom In and Out
        elif event.type == pygame.MOUSEWHEEL:
            # Scroll up (event.y = 1) zooms in, Scroll down (event.y = -1) zooms out
            cam_distance -= event.y * 1.5
            
            # Set boundaries so you can't zoom inside the black hole or too far away
            cam_distance = max(4.0, min(40.0, cam_distance))

    # Pass the running time AND the zoom distance into the render engine
    frame_buffer = render_frame(sim_time, cam_distance)

    surface = pygame.surfarray.make_surface(np.transpose(frame_buffer, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)
    
pygame.quit()
sys.exit()