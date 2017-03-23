/*
Heat flow in a 2D plate, via (silly!) iterative method.

This is a sequential program, but it can be made parallel
by changing the main function.

Orion Sky Lawlor, olawlor@acm.org, 2008-04-01 (Public Domain)
*/
#include <iostream>
#include <fstream>
using std::fstream;
#include <vector>
#include <algorithm> /* for std::swap */
#include <stdlib.h> /* for atoi */
#include <chrono>

//including omp.h to access OpenMP library//
#include <omp.h>
/**
  A 2D mesh, including both interior data and exterior boundaries.
*/
class mesh2d {
public:
	/* Size of our interior data, not counting one row of boundaries */
	int wid,ht;
	mesh2d(int w,int h) {allocate(w,h);}
	/* Allocate our interior data to be w x h in size */
	void allocate(int w,int h) {
		row_size=w+2; /* include boundaries */
		wid=w;
		ht=h;
		data.resize(row_size*(ht+2),1.0e-10); /*<- avoid roundoff! */
		interior=&data[1+row_size*1]; /* our first real pixel */
		nextdata.resize(row_size*(ht+2),1.0e-10);
		nextinterior=&nextdata[1+row_size*1]; /* our first real pixel */
	}
	
	/* Return a pointer to our i'th row.  
		i==-1 is our first boundary row.
		i==0 is our first interior row.  
		i==ht-1 is our last interior row.
		i==ht is our last boundary row.
	   The returned row[-1] and row[wid] are boundary data.
	*/
	float *row(int i) {return &interior[0+row_size*i];}
	
	/* Fill our boundaries with problem-domain data */
	void set_boundaries(void);
	/* Compute data in our interior.  Boundaries must be set first. */
	void compute(void);
private:
	int row_size; /* *total* data elements per row (counting boundaries) */
	/* Pixel data access pointers. These point to the first *interior* pixel. */
	float *interior; /* <- real data */
	float *nextinterior; /* <- this version *only* used inside "compute" */
	/* Pixel data storage (never used) */
	std::vector<float> data; 
	std::vector<float> nextdata; 
};

/* Fill our boundaries with problem-domain data */
void mesh2d::set_boundaries(void) {
	int x,y;
	x=-1;  for (y=-1;y<=ht;y++) row(y)[x]=0.0; /* left */
	x=wid; for (y=-1;y<=ht;y++) row(y)[x]=1.0; /* right */
	y=-1;  for (x=0;x<wid;x++) row(y)[x]=0.5; /* top */
	y=ht;  for (x=0;x<wid;x++) row(y)[x]=1.0; /* bottom */
}
/* Compute data in our interior.  Boundaries must be set first. */
void mesh2d::compute(void) {
	int x,y;
	for (y=0;y<ht;y++) {
		float *p=row(y-1); /* previous row */
		float *c=row(y);   /* current row */
		float *n=row(y+1); /* next row */
		for (x=0;x<wid;x++) 
		{ /* our new value is the average of our four neighbors */
			nextinterior[x+y*row_size]=
				0.25*(p[x]+c[x-1]+c[x+1]+n[x]);
		}
	}
	/* OK, "nextinterior" now contains the new values. 
	   Swap them for the current values. */
	std::swap(nextinterior,interior);
}


int main(int argc, char **argv) {
	int n_steps = 1000;
	if (argc > 1) n_steps = atoi(argv[1]);
	fstream myfile;
	myfile.open("resultfileOpenMP.txt", std::ios::out);


	/* Set up a mesh (uncomment exactly one of these sizes) */
	//int w=80, h=40; /* small mesh (just 10us/step) */
	//int w=800, h=400;  /* big mesh (4ms/step) */
	int w = 2400, h = 1200; /* huge mesh (42ms/step) */
	mesh2d m(w, h);

	// Timing Code
	std::chrono::time_point <std::chrono::high_resolution_clock> Total, TotalEnd;
	std::chrono::time_point <std::chrono::high_resolution_clock> Mesh, MeshEnd;
	std::chrono::time_point <std::chrono::high_resolution_clock> MeshPPM, MeshPPMEnd;
	std::chrono::time_point <std::chrono::high_resolution_clock> PPM, PPMEnd;
	Total = std::chrono::high_resolution_clock::now();
	Mesh = std::chrono::high_resolution_clock::now();
	MeshPPM = std::chrono::high_resolution_clock::now();
	
	/* Compute data on the mesh for a series of timesteps */

#pragma omp parallel for num_threads(8)
		for (int timestep = 0; timestep < n_steps; timestep++) {
			if ((timestep % 1000) == 0) (std::cout << "Starting step " << timestep << "\n").flush();
			m.set_boundaries();
			m.compute();
		}

	MeshEnd = std::chrono::high_resolution_clock::now();
	PPM = std::chrono::high_resolution_clock::now();
	/* Write out the final heat data as a PPM image */
	printf("Writing out %dx%d pixel image\n",w,h);
	std::ofstream out("out.ppm",std::ios_base::binary);
	out<<"P6\n"
	   <<w<<" "<<h<<"\n"
	   <<"255\n";
#pragma omp parallel for num_threads(8)
	for (int y=0;y<h;y++)
		for (int x=0;x<w;x++) {
		unsigned char c=(unsigned char)(255*m.row(y)[x]);
		for (int channel=0;channel<3;channel++) /* RGB color channels */
			out.write((char *)&c,1);
	}
	
	TotalEnd = std::chrono::high_resolution_clock::now();
	MeshPPMEnd = std::chrono::high_resolution_clock::now();
	PPMEnd = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> Total_elapsed_seconds = TotalEnd - Total;
	std::chrono::duration<double> Mesh_elapsed_seconds = MeshEnd - Mesh;
	std::chrono::duration<double> MeshPPM_elapsed_seconds = MeshPPMEnd - MeshPPM;
	std::chrono::duration<double> PPM_elapsed_seconds = PPMEnd - PPM;

	myfile << "Total Time: " << Total_elapsed_seconds.count() << "seconds.\n\n";
	std::cout << "Total Time:" << Total_elapsed_seconds.count() << " seconds.\n\n";
	
	myfile << "Time for Mesh+PPM Image computations: "<< MeshPPM_elapsed_seconds.count() << "seconds.\n\n";
	std::cout << "Total Time: " << MeshPPM_elapsed_seconds.count() << "seconds.\n\n";
	
	myfile << "Time for Mesh computations: " << Mesh_elapsed_seconds.count() << "seconds.\n\n";
	std::cout << "Total Time: " << Mesh_elapsed_seconds.count() << "seconds.\n\n";

	myfile << "Time for PPM Image computations: " << PPM_elapsed_seconds.count() << "seconds.\n\n";
	std::cout << "Total Time: " << PPM_elapsed_seconds.count() << "seconds.\n\n";

	myfile.close();
	return 0;
}
