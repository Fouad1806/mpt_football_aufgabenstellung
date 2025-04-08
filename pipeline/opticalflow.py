class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!

    def start(self, data):
        """
        when function is called, write current image into class for later calculation
        """
        # ToDo: Implement start up procedure of the module
        self.prev_img = data["image"]
        pass

    def stop(self, data):
        # ToDo: Implement shut down procedure of the module
        self.prev_img = None # reset prev_img stored
        pass

    def step(self, data):
        """
        `data` parameter hands over all signals from previous modules
        
        important note: results must be embedded into a python dictionary
        - for opticalFlow the dict entry is a 1x2 numpy array with delta_x and delta_y shifts
            - this is important for the visual overlay over the video clip!

        image array shape: (540, 960, 3) [height, width, channels]

        ## Theoretical background:
        from the lectures:
        $$\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} \Sigma_i I^2_{xi} & \Sigma_i I_{xi} I_{yi} \\ \Sigma_i I_{xi} I_{yi} & \Sigma_i I^2_{yi} \end{pmatrix}^{-1} \cdot \begin{pmatrix}-\Sigma_i I_{xi} I_{ti} \\ -\Sigma_i I_{yi} I_{ti} \end{pmatrix}\$$
        
        this system of linear equations should be solved for small image regions

        ## Approach:
        This calculation compares 2 images, so it can only start on frame 2!
        1. when opticalFlow is initialized, save current image for calculations for the next frame 
        2. we're working with image regions, so we need to select an arbitrary selection
            - `20px x 20px` for now, because that allows us to use squares for the selection while having equal size sections over the entire image
        3. we'll calculate the optical flow using the system of linear equations described above
        4. overwrite previous image with current image for next calculation
        5. we'll average the resulting optical flow vectors and return them to the dictionary
            with the selected regions that will be the average `(delta_x, delta_y)` for 1296 regions
        
        ## Possible problems:
        - due to parallax the average optical flow close to the camera should be significantly less than at the far end of the image
            if the camera is not in top-down perspective!
            - could allow for averaging row wise, to calculate the optical flow compensation for different depths if the geometry of the scene is known!
        
        ## Questions:
        verstehe nicht wie das funktionieren soll wenn len(x) =! len(y), wie bei dem Bild?
        """
        # ToDo: Implement processing of a single frame
        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        # You 

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector


        # get current and previous image
        y, x, channels = data["image"].shape
        prev_img = self.prev_img

        # prep for image region
        upper_sigma_lim = 20

        # \Sigma_i I^2_{xi}
        sum_1 = lambda I: sum(x**2 for x in I)
        # \Sigma_i I_{xi} I_{yi}
        sum_2 = lambda I: sum()
        # \Sigma_i I_{xi} I_{yi}
        
        # \Sigma_i I^2_{yi} \end{pmatrix}^{-1}
        
        # -\Sigma_i I_{xi} I_{ti}
        
        # -\Sigma_i I_{yi} I_{ti} 

        
        self.prev_img = data["image"] # overwrite for next calculation

        return {
           "opticalFlow": None
        } 
