
#
#              _____                                                         
#              __  /_____________ _____  ____________________                
#              _  __/_  ___/  __ `/_  / / /  ___/  _ \_  ___/                
#              / /_ _  /   / /_/ /_  /_/ // /__ /  __/  /                    
#              \__/ /_/    \__,_/ _\__, / \___/ \___//_/                     
#                                 /____/                                     
#    
#                 a simple ray tracer written in python                                                                              
#                    https://github.com/bpops/traycer
#

import array
from   tqdm    import tqdm
import numpy as np
import math
import random

class ppm:
    """
    PPM Image
    """

    def __init__(self, width=512, height=512, maxval=255):
        """
        Initialize a new PPM image

        Parameters
        ----------
        width : int, optional
            width in pixels (default: 512)
        height : int, optional
            height in pixels (default: 512)
        maxval : int, optional
            maximum color value (default: 255)
        """
        self.width  = width
        self.height = height
        self.maxval = maxval
        self.image  = array.array('B', [0, 100, 255] * width * height)

    def write_color(self, x, y, rgb):
        """
        Write the color of a pixel

        Parameters
        ----------
        x : int
            x coordinate
        y : int
            y coordinate
        rgb : int
            RGB color value
        """
        idx = int((y * self.width + x) * 3)
        self.image[idx:idx+3] = array.array('B', rgb)

    def gradient_test(self):
        """
        Test generate a color gradient
        """
        for i in tqdm(range(self.width), position=0):
            for j in tqdm(range(self.height), position=1, leave=False):
                rgb = color(i / (self.width-1), j / (self.height-1), 0.25)
                self.write_color(i, j, rgb.tuple())

    def write(self, filename):
        """
        Write the image to a file

        Parameters
        ----------
        filename : str
            filename to write tos
        """
        ppm_header = f"P6 {self.width} {self.height} {self.maxval}\n"
        with open(filename, 'wb') as f:
            f.write(bytearray(ppm_header, 'ascii'))
            self.image.tofile(f)

class vec3:
    """
    3D Vector
    """

    def __init__(self, x=0, y=0, z=0):
        """
        Initialize a new vector

        Parameters
        ----------
        x : int
            x component
        y : int
            z component
        z : int
            z component
        """
        self.x = x
        self.y = y
        self.z = z
        
    def length(self):
        """
        Return the length of the vector

        Returns
        -------
        out : float
            the length of the array
        """
        return math.sqrt(self.length_squared())
    
    def length_squared(self):
        """
        Return the length of the vector squared

        Returns
        -------
        out : float
            the length of the array, squared
        """
        return self.x*self.x + self.y*self.y + self.z*self.z
    
    def __add__(self, other):
        """
        Add two vectors

        Returns
        -------
        out : vec3
            vector created by addition of two vectors
        """
        return self.__class__(self.x + other.x, self.y + other.y, self.z +
                              other.z)
    
    def __sub__(self, other):
        """
        Subtract two vectors

        Returns
        -------
        out : vec3
            vector created by subtraction of two vectors
        """
        return self.__class__(self.x - other.x, self.y - other.y, self.z -
                              other.z)
    
    def __mul__(self, other):
        """
        Multiply a vector by a scalar or another vector

        Returns
        -------
        out : vec3
            resulting vector
        """
        if type(other) is vec3:
            return self.__class__(self.x * other.x, self.y * other.y, 
                                  self.z * other.z)
        else:
            return self.__class__(self.x * other, self.y * other, 
                                  self.z * other)

    def __rmul__(self, other):
        """
        Multiply a vector by a scalar or another vector

        Returns
        -------
        out : vec3
            resulting vector
        """
        if type(other) is vec3:
            return self.__class__(self.x * other.x, self.y * other.y, 
                                  self.z * other.z)
        else:
            return self.__class__(self.x * other, self.y * other, 
                                  self.z * other)

    def __truediv__(self, val):
        """
        Divide a vector by a scalar

        Returns
        -------
        out : vec3
            vector created by division of two vectors
        """
        return self.__class__(self.x / val, self.y / val, self.z / val)

    def dot(self, other):
        """
        Return the dot product of two vectors

        Returns
        -------
        out : vec3
            vector created by dot product of two vectors
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """
        Return the cross product of two vectors

        Returns
        -------
        out : vec3
            vector created by cross product of two vectors
        """
        return vec3(self.y * other.z - self.y * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)

    def unit_vector(self):
        """
        Return a unit vector in the same direction as this vector

        Returns
        -------
        out : vec3
            unit vectors
        """
        return self / self.length()

    def tuple(self):
        """
        Return the vector as a tuple

        Returns
        -------
        out : array_like
            x, y, and z components of the vector
        """
        return np.asarray((self.x, self.y, self.z))
    
    def randomize(self, min_v=0.0, max_v=1.0):
        """
        Randomizes the vector

        Parameters
        ----------
        min_v : float
            minimum value (default 0.0)
        max_v : float
            maximum value (default 1.0)
        """
        self.x = np.random.uniform(min_v,max_v)
        self.y = np.random.uniform(min_v,max_v)
        self.z = np.random.uniform(min_v,max_v)

def random_in_unit_sphere():
    """
    Get random vector in unit sphere
    """
    while True:
        p = vec3()
        p.randomize(-1,1)
        if p.length_squared() < 1:
            return p

def random_unit_vector():
    """
    Get random vector on the unit sphere
    """
    return random_in_unit_sphere().unit_vector()

def random_on_hemisphere(normal):
    on_unit_sphere = random_unit_vector()
    if on_unit_sphere.dot(normal) > 0.0: # same hemisphere
        return on_unit_sphere
    else:
        return -1*on_unit_sphere

class color(vec3):
    """
    Color
    """
    
    def __init__(self, r=0, g=0, b=0):
        """
        Initialize a new color vector

        Parameters
        ----------
        r : int
            red color value
        g : int
            green color value
        b : int
            blue color value
        """
        super().__init__(r, g, b)
        
    def write_color(self):
        """
        Write the color vector to a string

        Returns
        -------
        out : str
            string of RGB colors
        """
        r = int(255.999 * self.x)
        g = int(255.999 * self.y)
        b = int(255.999 * self.z)
        return f"{r} {g} {b}"
    
    def tuple(self):
        """
        Returns a tuple of the RGB colors

        Returns
        -------
        out : array_like
            tuple of RGB color values
        """
        return np.asarray((super().tuple()*255.999), dtype=np.int32)


class point3(vec3):
    """
    3D Point
    """

    def __init__(self, x=0, y=0, z=0):
        """
        Initialize a new point

        Parameters
        ----------
        x : float
            x location
        y : float
            y location
        z : float
            z location
        """
        super().__init__(x, y, z)

class ray:
    """
    Tracing Ray
    """

    def __init__(self, origin, direction):
        """
        Initialize a new ray

        Parameters
        ----------
        origin : point3
            origin coordinates
        direction : point3
            ray direction
        """
        self.origin    = origin
        self.direction = direction

    def at(self, t) -> point3:
        """
        Return the point along the ray at distance t

        Returns
        -------
        out : float
            point along ray at distance t
        """
        return self.origin + self.direction * t
    
"""
def ray_color(r):
    unit_direction = r.direction.unit_vector()
    a = 0.5 * (unit_direction.y + 1.0)
    return (1.0-a) * color(1.0,1.0,1.0) + a * color(0.5,0.7,1.0)
"""


class interval:
    """
    Interval
    """

    def __init__(self, min_t=-1*np.inf, max_t=np.inf):
        self.min = min_t
        self.max = max_t

    def contains(self, x):
        """
        Contains

        Parameters
        x : float
            number to check
        """
        return (self.min <= x) and (x <= self.max)
    
    def surrounds(self, x):
        """
        Surrounds

        Parameters
        x : float
            number to check
        """
        return (self.min < x) and (x < self.max)
    
    def empty(self):
        """
        Empty the interval
        """
        self.min = np.inf
        self.max = -1*np.inf

    def universe(self):
        """
        Set interval to universe
        """
        self.min = -1*np.inf
        self.max = -np.inf    

    def clamp(self, x):
        """
        Clamp
        """
        if x < self.min:
            return self.min
        if x > self.max:
            return self.max
        return x

class camera():
    """
    Camera
    """

    def __init__(self, aspect_ratio=16.0/9.0, image_width=500, focal_length=1.0,
                 viewport_height=2.0, center=point3(0,0,0)):
        """
        Initialize camera

        Parameters
        ----------    
        aspect_ratio : float
            aspect ratio (default 16/9)
        image_width : int
            image width (default 500)
        focal_length : float
            focal length (default 1.0)
        viewport_height : float
            viewport height (default 2.0)
        center : point3
            camera center (default (0,0,0))
        """
        self.aspect_ratio = aspect_ratio
        self.image_width  = image_width
        self.image_height = int(image_width/aspect_ratio)

        # camera
        self.focal_length = 1.0
        self.viewport_height = 2.0
        self.viewport_width = self.viewport_height * self.image_width / self.image_height
        self.center = point3(0,0,0)

        # calculate the vectors across horizontal and down vertical viewport edges
        self.viewport_u = vec3(self.viewport_width, 0, 0)
        self.viewport_v = vec3(0, -1*self.viewport_height, 0)

        # calculate the horizontal and vertical delta vectors from pixel to pixel
        self.pixel_delta_u = self.viewport_u / self.image_width
        self.pixel_delta_v = self.viewport_v / self.image_height

        # calculate the location of the upper left pixel
        self.viewport_upper_left = self.center - vec3(0, 0, self.focal_length) - \
            self.viewport_u/2 - self.viewport_v/2
        self.pixel00_loc = self.viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)


    def render(self, world, aa=False, max_depth=10):
        """
        Render image
        
        Parameters
        ----------
        world : hittable_list
            world to render
        aa : int
            samples per pixel for anti-aliasing (default 1)
        max_depth : int
            maximum  number of ray bounces into scene (default 10)
        
        Returns
        -------
        image : ppm
            rendered image
        """
        
        # generate pixel by pixel
        image = ppm(width=self.image_width, height=self.image_height)
        for j in tqdm(range(self.image_height), desc="Scanlines rendered"):
            for i in range(self.image_width):
                if not aa is False:
                    pixel_color = color(0,0,0)
                    for a in range(aa):
                        r = self.get_randray(i, j)
                        pixel_color += self.ray_color(r, max_depth, world)
                    pixel_color /= aa
                else:
                    pixel_center = self.pixel00_loc + (i*self.pixel_delta_u) + (j*self.pixel_delta_v)
                    ray_direction = pixel_center - self.center
                    r = ray(self.center, ray_direction)
                    pixel_color = self.ray_color(r, world)
                image.write_color(i, j, pixel_color.tuple())

        return image

    def ray_color(self, r, depth, world):
        """
        Determine ray color

        Parameters
        ----------
        r : ray
            tracing ray
        depth : int
            maximum number of ray bounces
        world : hittables_list
            list of hittables
        """
        if depth <= 0:
            return color(0,0,0)

        hit, rec = world.hit(r, ray_t=interval(0.001, np.inf))
        if hit:
            direction = random_on_hemisphere(rec.normal)
            return 0.5 * self.ray_color(ray(rec.p, direction), depth-1, world)
            #return 0.5 * self.ray_color(ray(rec.p, direction), world)
            #return 0.5 * (color(1,1,1) + rec.normal)

        unit_direction = r.direction.unit_vector()
        a = 0.5 * (unit_direction.y + 1.0)
        return color(1.0,1.0,1.0)*(1.0-a) + color(0.5,0.7,1.0)*a

    def pixel_sample_square(self):
        """
        Returns a random point in the square surrounding a pixel at origin
        """
        px = -0.5 + np.random.uniform(0.0,1.0)
        py = -0.5 + np.random.uniform(0.0,1.0)
        return (px * self.pixel_delta_u) + (py * self.pixel_delta_v)

    def get_randray(self, i, j):
        """
        Get Random Ray for pixel at location (i,j)
        
        Parameters
        ----------
        i : int
            horizontal pixel location
        j : int
            vertical pixel location

        Returns
        ray : ray
            random ray
        """

        pixel_center = self.pixel00_loc + (i*self.pixel_delta_u) + (j*self.pixel_delta_v)
        pixel_sample = pixel_center + self.pixel_sample_square()

        ray_origin = self.center
        ray_direction = pixel_sample - ray_origin
        return ray(ray_origin, ray_direction)

class hit_record():
    """
    Hit Record
    """

    def __init__(self):
        self.p = point3(0,0,0)
        self.normal = vec3(0,0,0)
        self.t = 0.0
        self.front_face = None

    def set_face_normal(self, r, outward_normal):
        """
        Set the hit record normal vector
        
        Parameters
        ----------
        r : ray
            ray
        outword_normal: vec3
            outward normal
        """

        outward_normal = outward_normal.unit_vector()
        self.front_face = r.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else -1*outward_normal

class hittable_list():
    """
    Hittable List
    """

    def __init__(self):
        """
        Initialize hittable list
        """
        self.clear()

    def add(self, object):
        """
        Add an object

        Parameters
        ----------
        object : hittable
            object to add
        """
        self.objects.append(object)

    def clear(self):
        """
        Clears objects
        """
        self.objects = []

    def hit(self, r, ray_t=interval(-1*np.inf, np.inf)):
        """"
        Hit
        
        Parameters
        ----------
        ray : ray
            tracing ray
        ray_tmin : float
            minimum t (default 0)
        ray_tmax : float
            maximum t (default infinity)
        rec: hit_record
            hit record

        Returns
        -------
        hit_anything : bool
            whether or not the ray hit anything
        rec : hit_record
            record of all hits
        """

        rec = hit_record()
        hit_anything = False
        closest_so_far = ray_t.max

        for object in self.objects:
            if object.hit(r, ray_t=interval(ray_t.min, closest_so_far),
                          rec=rec):
                hit_anything = True
                closest_so_far = rec.t
                #rec = temp_rec

        return hit_anything, rec

class hittable():
    """
    Hittable object
    """

    def __init__(self, center):
        self.center = center

    def hit(self, r, ray_t, hit_record):
        pass

class sphere(hittable):
    """
    Sphere
    """

    def __init__(self, center, radius):
        """
        Initialize a new sphere

        Parameters
        ----------
        center : point3
            center point of sphere
        radius : float
            radius of sphere
        """
        super().__init__(center)
        self.radius = radius

    def hit(self, r, ray_t=interval(-1*np.inf, np.inf), rec=None):
        """
        Determine sphere color based on ray

        Parameters
        ----------
        ray : vec3
            3d vector of ray
        ray_tmin : float
            minimum t
        ray_tmax : float
            maximum t
        rec: hit_record
            hit record
        
        Returns
        -------
        out : float
            color value
        """
        oc = r.origin - self.center
        a = r.direction.length_squared()
        half_b = oc.dot(r.direction)
        c = oc.length_squared() - self.radius**2

        discriminant = half_b*half_b - a*c
        if discriminant < 0:
            return False
        sqrtd = math.sqrt(discriminant)

        # find the nearest root that lies in the acceptable range
        root = (-half_b - sqrtd)  / a
        if not ray_t.surrounds(root):
            root = (-half_b + sqrtd) / a
            if not ray_t.surrounds(root):
                return False
        #if (root <= ray_tmin) or (ray_tmax <= root):
        #    root = (-half_b + sqrtd) / a
        #    if (root <= ray_tmin) or (ray_tmax <= root):
        #        return False
            
        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(r, outward_normal)
        #rec.normal = (rec.p - self.center) / self.radius
 
        return True

        """
        if discriminant < 0:
            return -1;
        else:
            return (-half_b - math.sqrt(discriminant)) / a
        """
        
    """
    def ray_color(self, r):
        t = self.hit(r)
        if t > 0.0:
            N = (r.at(t)-vec3(0,0,-1)).unit_vector()
            return 0.5*color(N.x+1, N.y+1, N.z+1)
        else:
            unit_direction = r.direction.unit_vector()
            a = 0.5*(unit_direction.y+1.0)
            return (1.0-a)*color(1.0,1.0,1.0) + a*color(0.5,0.7,1.0)
    """

