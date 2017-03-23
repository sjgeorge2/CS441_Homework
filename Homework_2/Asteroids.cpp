#include <iostream>
#include <fstream>
using std::fstream;
using std::ios;
#include <chrono> //c++11 Timing functions
#include <cmath>
//#include "lib/inc.c" // netrun timing functions
#include "xmmintrin.h";

// Make up a random 3D vector in this range.
//   NOT ACTUALLY RANDOM, just pseudorandom via linear congruence.
void randomize(int index, float range, float &x, float &y, float &z) {
	index = index ^ (index << 24); // fold index (improve whitening)
	x = (((index * 1234567) % 1039) / 1000.0 - 0.5)*range;
	y = (((index * 7654321) % 1021) / 1000.0 - 0.5)*range;
	z = (((index * 1726354) % 1027) / 1000.0 - 0.5)*range;
}

class position {
public:
	float px, py, pz; // position's X, Y, Z components (meters)

					  // Return distance to another position
	float distance(const position &p) const {
		float dx = p.px - px;
		float dy = p.py - py;
		float dz = p.pz - pz;
		return sqrt(dx*dx + dy*dy + dz*dz);
	}
};

class body : public position {
public:
	float m; // mass (Kg)
};

class asteroid : public body {
public:
	float vx, vy, vz; // velocity (m)
	float fx, fy, fz; // net force vector (N)

	void setup(void) {
		fx = fy = fz = 0.0;
	}

	// Add the gravitational force on us due to this body
	void add_force(const body &b) {
		// Newton's law of gravitation:
		//   length of F = G m1 m2 / r^2
		//   direction of F = R/r
		float dx = b.px - px;
		float dy = b.py - py;
		float dz = b.pz - pz;
		float r = sqrt(dx*dx + dy*dy + dz*dz);

		float G = 6.67408e-11; // gravitational constant
		float scale = G*b.m*m / (r*r*r);
		fx += dx*scale;
		fy += dy*scale;
		fz += dz*scale;
	}

	// Use known net force values to advance by one timestep
	void step(float dt) {
		float ax = fx / m, ay = fy / m, az = fz / m;
		vx += ax*dt; vy += ay*dt; vz += az*dt;
		px += vx*dt; py += vy*dt; pz += vz*dt;
	}
};

// A simple fixed-size image
class image {
public:
	enum { pixels = 500 };
	unsigned char pixel[pixels][pixels];
	void clear(void) {
		for (int y = 0; y<pixels; y++)
			for (int x = 0; x<pixels; x++)
				pixel[y][x] = 0;
	}

	void draw(float fx, float fy) {
		int y = (int)(fx*pixels);
		int x = (int)(fy*pixels);
		if (y >= 0 && y<pixels && x >= 0 && x<pixels)
			if (pixel[y][x]<200) pixel[y][x] += 10;
	}

	void write(const char *filename) {
		std::ofstream f("out.ppm", std::ios_base::binary);
		f << "P5 " << pixels << " " << pixels << "\n";
		f << "255\n";
		for (int y = 0; y<pixels; y++)
			for (int x = 0; x<pixels; x++)
				f.write((char *)&pixel[y][x], 1);
	}
};

int main(void) {
	image img;
	fstream myfile;

	enum { n_asteroids = 8192 };
	float range = 500e6;
	float p2v = 3.0e-6; // position (meters) to velocity (meters/sec)

	body terra;
	terra.px = 0.0; terra.py = 0.0; terra.pz = 0.0;
	terra.m = 5.972e24;

	body luna;
	luna.px = 384.4e6; luna.py = 0.0; luna.pz = 0.0;
	luna.m = 7.34767309e22;

	//the c++11 timing code
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	for (int test = 0; test < 5; test++) {
		float closest_approach = 1.0e100;
		img.clear(); // black out the image


		start = std::chrono::high_resolution_clock::now();
		//double start=time_in_seconds();
		/* performance critical part here */
		for (int ai = 0; ai < n_asteroids; ai++)
		{
			asteroid a;
			int run = 0;

			do {

				randomize(ai * 100 + run, range, a.px, a.py, a.pz);
				run++;
			} while (a.distance(terra) < 10000e3);



			a.m = 1.0;
			a.vx = -a.py*p2v; 
			a.vy = a.px*p2v; 
			a.vz = 0.0;
		

			for (int i = 0; i < 1000; i++)
			{

				a.setup();
				a.add_force(terra);
				a.add_force(luna);
				a.step(1000.0);

				// Draw current location of asteroid
				img.draw(
					a.px*(1.0 / range) + 0.5,
					a.py*(1.0 / range) + 0.5);

				// Check distance
				float d = terra.distance(a);
				if (closest_approach > d) closest_approach = d;
			}
		}
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Took " << elapsed_seconds.count() << " seconds, " << elapsed_seconds.count()*1.0e9 / n_asteroids << " ns/asteroid\n";
		std::cout << "  closest approach: " << closest_approach << "\n";

		myfile.open("resultfile.txt", ios::out);
		if (myfile.is_open())
		{
			myfile << "Unaltered time:\n";
			myfile << "Took " << elapsed_seconds.count() << " seconds, " << elapsed_seconds.count()*1.0e9 / n_asteroids << " ns/asteroid\n";
			myfile << "  closest approach: " << closest_approach << "\n";
		}
	}
///////////////////////////////////////////////////////////////////////
					/* the SSE timing code */
//////////////////////////////////////////////////////////////////////
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	for (int test = 0; test < 5; test++) {
		float closest_approach = 1.0e100;
		img.clear(); // black out the image


		start = std::chrono::high_resolution_clock::now();
		//double start=time_in_seconds();
		/* performance critical part here */
		// add SSE instructions here //
		
		for (int ai = 0; ai < n_asteroids; ai = ai+4)
		{
			// create 4 instances of asteroids
			// "simulating" parallelism by doing four operations at once //
			// Goal is to change this to SSE registers and operate on them //
			asteroid a;

			int run = 0;
			do {
				// run randomize on each SSE
				randomize(ai * 100 + run, range, a.px, a.py, a.pz);
				run++;

			} while (a.distance(terra) < 10000e3);

			//a.m = 1.0;
			//a.vx = -a.py*p2v; 
			//a.vy = a.px*p2v; 
			//a.vz = 0.0;
			// load one SSE register full of p2v
			// load second SSE register full of components of an asteroid, to be put in a vector to set each component with
			fourfloats src(1.0, -a.py, a.px, 0.0);
			fourfloats d = (src, src*p2v, src*p2v, src);
			float dest[4];
			d.store(dest);
			a.m = dest[0];
			a.vx = dest[1];
			a.vy = dest[2];
			a.vz = dest[3];

			for (int i = 0; i < 1000; i++)
			{

				a.setup();
				a.add_force(terra);	
				a.add_force(luna);
				a.step(1000.0);

				// Draw current location of asteroid
				img.draw(
					a.px*(1.0 / range) + 0.5,
					a.py*(1.0 / range) + 0.5);

				// Check distance
				float d = terra.distance(a);
				if (closest_approach > d) closest_approach = d;
			}
		}
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Took " << elapsed_seconds.count() << " seconds, " << elapsed_seconds.count()*1.0e9 / n_asteroids << " ns/asteroid\n";
		std::cout << "  closest approach: " << closest_approach << "\n";

		myfile.open("resultfile.txt", ios::out);
		if (myfile.is_open())
		{
			myfile << "SSE Altered time:\n";
			myfile << "Took " << elapsed_seconds.count() << " seconds, " << elapsed_seconds.count()*1.0e9 / n_asteroids << " ns/asteroid\n";
			myfile << "  closest approach: " << closest_approach << "\n";
		}
		myfile.close();
	}


	img.write("out.ppm"); // netrun shows "out.ppm" by default
}