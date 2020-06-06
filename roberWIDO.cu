/*
 * Simplified simulation of life evolution
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2019/2020
 *
 * v1.5
 *
 * CHANGES:
 * 1) Float values have been substituted by fixed point arithmetics 
 *	using integers. To simplify, the fixed point arithmetics are done 
 *	with PRECISION in base 10. See precision constant in int_float.h
 * 2) It uses a portable approximation to trigonometric functions using
 *	Taylor polynomials. 
 * 3) nrand48 function has been extracted from glibc source code and 
 *	its internal API simplified to allow its use in the GPU.
 *
 * (c) 2020, Arturo Gonzalez Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <cstring>
#include<math.h>
#include<stdbool.h>
#include<cputils.h>
#include<cuda.h>
#include<int_float.h>

/* 
 * Constants: Converted to fixed point with the given PRECISION
 */
#define ENERGY_NEEDED_TO_LIVE		PRECISION / 10	// Equivalent to 0.1
#define ENERGY_NEEDED_TO_MOVE		PRECISION	// Equivalent to 1.0
#define ENERGY_SPENT_TO_LIVE		PRECISION / 5	// Equivalent to 0.2
#define ENERGY_SPENT_TO_MOVE		PRECISION	// Equivalent to 1.0
#define ENERGY_NEEDED_TO_SPLIT		PRECISION * 20	// Equivalent to 20.0

__device__ __constant__ int columnsDev;
/* Structure to store data of a cell */
typedef struct {
	int pos_row, pos_col;		// Position
	int mov_row, mov_col;		// Direction of movement
	int choose_mov[3];		// Genes: Probabilities of 0 turning-left; 1 advance; 2 turning-right
	int storage;			// Food/Energy stored
	int age;			// Number of steps that the cell has been alive
	unsigned short random_seq[3];	// Status value of its particular random sequence
	bool alive;			// Flag indicating if the cell is still alive
} Cell;


/* Structure for simulation statistics */
typedef struct {
	int history_total_cells;	// Accumulated number of cells created
	int history_dead_cells;		// Accumulated number of dead cells
	int history_max_alive_cells;	// Maximum number of cells alive in a step
	int history_max_new_cells;	// Maximum number of cells created in a step
	int history_max_dead_cells;	// Maximum number of cells died in a step
	int history_max_age;		// Maximum age achieved by a cell
	int history_max_food;		// Maximum food level in a position of the culture
} Statistics;


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 *	USE THIS SPACE FOR YOUR KERNEL OR DEVICE FUNTIONS
 *
 */

#include "taylor_trig.h"
#include "glibc_nrand48.h"

/*
 * Get an uniformly distributed random number between 0 and max
 * It uses glibc_nrand, that returns a number between 0 and 2^31
 */
#define int_urand48( max, seq )	(int)( (long)(max) * glibc_nrand48( seq ) / 2147483648 )

/* 
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be modified by the students if needed
 *
 */
#define accessMat( arr, exp1, exp2 )	arr[ (int)(exp1) * columns + (int)(exp2) ]

/*
 * Function: Choose a new direction of movement for a cell
 * 	This function can be changed and/or optimized by the students
 */
__device__ void cell_new_direction( Cell *cell ) {
	int angle = int_urand48( INT_2PI, cell->random_seq );
	cell->mov_row = taylor_sin( angle );
	cell->mov_col = taylor_cos( angle );
}

void cell_new_direction2( Cell *cell ) {
	int angle = int_urand48( INT_2PI, cell->random_seq );
	cell->mov_row = taylor_sin( angle );
	cell->mov_col = taylor_cos( angle );
}

/*
 * Function: Mutation of the movement genes on a new cell
 * 	This function can be changed and/or optimized by the students
 */

//poner __device__ delante y asi se le puede llamar desde una función __global__
__device__ void cell_mutation( Cell *cell ) {
	/* 1. Select which genes change:
	 	0 Left grows taking part of the Advance part
	 	1 Advance grows taking part of the Left part
	 	2 Advance grows taking part of the Right part
	 	3 Right grows taking part of the Advance part
	*/
	int mutation_type = int_urand48( 4, cell->random_seq );
	/* 2. Select the amount of mutation (up to 50%) */
	int mutation_percentage = int_urand48( PRECISION / 2, cell->random_seq );
	/* 3. Apply the mutation */
	int mutation_value;
	switch( mutation_type ) {
		case 0:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[0] += mutation_value;
			break;
		case 1:
			mutation_value = intfloatMult( cell->choose_mov[0] , mutation_percentage );
			cell->choose_mov[0] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 2:
			mutation_value = intfloatMult( cell->choose_mov[2] , mutation_percentage );
			cell->choose_mov[2] -= mutation_value;
			cell->choose_mov[1] += mutation_value;
			break;
		case 3:
			mutation_value = intfloatMult( cell->choose_mov[1] , mutation_percentage );
			cell->choose_mov[1] -= mutation_value;
			cell->choose_mov[2] += mutation_value;
			break;
	}
	/* 4. Correct potential precision problems */
	cell->choose_mov[2] = PRECISION - cell->choose_mov[1] - cell->choose_mov[0];
}

/*
 * CUDA block reduction
 * Inputs: 
 *	Device pointer to an array of int of any size
 *	Size of the array
 *	Device pointer to an int to store the result
 * 
 * Launching parameters:
 *	One-dimesional grid of any size
 *	Any valid block size
 *	Dynamic shared memory size equal to: sizeof(int) * block size
 *
 * (c) 2020, Arturo Gonzalez-Escribano
 * Simplification for an assignment in a Parallel Computing course,
 * Computing Engineering Degree, Universidad de Valladolid
 * Academic year 2019/2020
 */


__global__ void reductionMax(int* array, int size, int* result, int *array2, int *numCells, int *stepNewCells, int* aDevice, Statistics* stats, int* stepDeadCellsDevice, int* numAliveDevice)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos==0){
		numCells[0] += stepNewCells[0];
	}
	extern __shared__ int buffer[ ];
	if ( globalPos < size ) { 
		array[globalPos] -= array[globalPos] / 20;
		array2[globalPos] = 0;
		buffer[ threadIdx.x ] = array[ globalPos ];
	}
	else buffer[ threadIdx.x ] = 0.0f;
	__syncthreads();

	for( int step=blockDim.x/2; step>=1; step /= 2 ) {
		if ( threadIdx.x < step )
			if ( buffer[ threadIdx.x ] < buffer[ threadIdx.x + step ] )
				buffer[ threadIdx.x ] = buffer[ threadIdx.x + step ];
		if ( step > 32 )
			__syncthreads();
	}

	if ( threadIdx.x == 0 )
		atomicMax( result, buffer[0] );

	if(globalPos==0){
		// Statistics: Max new cells per step
		if ( stepNewCells[0] > stats[0].history_max_new_cells ) stats[0].history_max_new_cells = stepNewCells[0];
		// Statistics: Accumulated dead and Max dead cells per step
		stats[0].history_dead_cells += stepDeadCellsDevice[0];
		if ( stepDeadCellsDevice[0] > stats[0].history_max_dead_cells ) stats[0].history_max_dead_cells = stepDeadCellsDevice[0];
		// Statistics: Max alive cells per step
		if ( numAliveDevice[0] > stats[0].history_max_alive_cells ) stats[0].history_max_alive_cells = numAliveDevice[0];
	}	
	if(globalPos==0){
		aDevice[globalPos] = 0;
		stepDeadCellsDevice[globalPos] = 0;
		stepNewCells[globalPos] = 0;
	}
}

__device__ void reductionMax2(Cell* array, int size, int *result)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	extern __shared__ int buffer[ ];
	if ( globalPos < size ) { 
		buffer[ threadIdx.x ] = array[ globalPos ].age;
	}
	else buffer[ threadIdx.x ] = 0.0f;
	__syncthreads();

	for( int step=blockDim.x/2; step>=1; step /= 2 ) {
		if ( threadIdx.x < step )
			if ( buffer[ threadIdx.x ] < buffer[ threadIdx.x + step ] )
				buffer[ threadIdx.x ] = buffer[ threadIdx.x + step ];
		if ( step > 32 )
			__syncthreads();
	}

	if ( threadIdx.x == 0 )
		atomicMax( result, buffer[0] );
}

__global__ void inicializarCelulas(Cell* cells, int size, int rows, int columns)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < size){
		cells[globalPos].alive = true;
		cells[globalPos].age = 1 + int_urand48( 19, cells[globalPos].random_seq );
		cells[globalPos].storage = 10 * PRECISION + int_urand48( 10 * PRECISION, cells[globalPos].random_seq );
		cells[globalPos].pos_row = int_urand48( rows * PRECISION, cells[globalPos].random_seq );
		cells[globalPos].pos_col = int_urand48( columns * PRECISION, cells[globalPos].random_seq );
		cell_new_direction( &cells[globalPos] );
		cells[globalPos].choose_mov[0] = PRECISION / 3;
		cells[globalPos].choose_mov[2] = PRECISION / 3;
		cells[globalPos].choose_mov[1] = PRECISION - cells[globalPos].choose_mov[0] - cells[globalPos].choose_mov[2];
	}
}

__global__ void repartoComidaFoodSpot(int* pos, int* food, int size, int columns, int* culture)
{
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < size){
		atomicAdd(&culture[pos[globalPos]], food[globalPos]);
	}
}

__global__ void limpiarCelulas(Cell* cells2, int* numCells, int* alive, Cell* cellsSal, int rows, int columns, int* array)
{	
	int pos = 0;
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < numCells[0] ) {
		if ( cells2[globalPos].alive ) {
			pos = atomicAdd(alive,1);
			int posicion = (cells2[globalPos].pos_row / PRECISION) * columns + (cells2[globalPos].pos_col / PRECISION);
			array[posicion] = 0;
			cellsSal[pos] = cells2[globalPos];
		}
	}
}

__global__ void limpiarCelulas2(Cell* cellsEntrada, int* alive, Cell* cellsSalida, int* numCells)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	int p = alive[0];
	if ( globalPos < p ) {
		cellsSalida[globalPos] = cellsEntrada[globalPos];
	}
	if(globalPos==0){
		numCells[0] = alive[0];
	}
}

__global__ void anyadirCelulas(int* size, Cell* newCells, Cell* cellsSal, int* numCells)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if ( globalPos < size[0] ) {
		cellsSal[ globalPos + numCells[0] ] = newCells[ globalPos ];
	}	
}

/*
__global__ void anyadirCelulas2(Cell* cellsEntrada, int* stepNewCells, int* numCells, Cell* cellsSalida)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < stepNewCells[0]){
		cellsSalida[numCells[0] + globalPos] = cellsEntrada[globalPos];
	}	
}*/

__global__ void final(int* cDevice, Statistics* stats)
{	
	if ( cDevice[0] > stats[0].history_max_food ) stats[0].history_max_food = cDevice[0];
	cDevice[0] = 0;
}

__global__ void movimientoCelulas(Cell* cellsEntrada, int rows, int columns, int* aliveDevice, int* stepDeadCellsDevice, Statistics* s, int* numCells, int* arrayCul, 
	int* arrayCulCells)
{	
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < numCells[0]){
		if ( cellsEntrada[globalPos].alive ) {
			cellsEntrada[globalPos].age ++;
			if ( cellsEntrada[globalPos].storage < ENERGY_NEEDED_TO_LIVE ) {
				cellsEntrada[globalPos].alive = false;
				atomicSub(aliveDevice,1);
				atomicAdd(stepDeadCellsDevice,1);
			}
			if ( cellsEntrada[globalPos].alive ) {
				if ( cellsEntrada[globalPos].storage < ENERGY_NEEDED_TO_MOVE ) {
					cellsEntrada[globalPos].storage -= ENERGY_SPENT_TO_LIVE;
				}
				else {
					cellsEntrada[globalPos].storage -= ENERGY_SPENT_TO_MOVE;
							
					int prob = int_urand48( PRECISION, cellsEntrada[globalPos].random_seq );
					if ( prob < cellsEntrada[globalPos].choose_mov[0] ) {
						int tmp = cellsEntrada[globalPos].mov_col;
						cellsEntrada[globalPos].mov_col = cellsEntrada[globalPos].mov_row;
						cellsEntrada[globalPos].mov_row = -tmp;
					}
					else if ( prob >= cellsEntrada[globalPos].choose_mov[0] + cellsEntrada[globalPos].choose_mov[1] ) {
						int tmp = cellsEntrada[globalPos].mov_row;
						cellsEntrada[globalPos].mov_row = cellsEntrada[globalPos].mov_col;
						cellsEntrada[globalPos].mov_col = -tmp;
					}
						
					cellsEntrada[globalPos].pos_row += cellsEntrada[globalPos].mov_row;
					cellsEntrada[globalPos].pos_col += cellsEntrada[globalPos].mov_col;

					if ( cellsEntrada[globalPos].pos_row < 0 ) cellsEntrada[globalPos].pos_row += rows * PRECISION;
					if ( cellsEntrada[globalPos].pos_row >= rows * PRECISION) cellsEntrada[globalPos].pos_row -= rows * PRECISION;
					if ( cellsEntrada[globalPos].pos_col < 0 ) cellsEntrada[globalPos].pos_col += columns * PRECISION;
					if ( cellsEntrada[globalPos].pos_col >= columns * PRECISION) cellsEntrada[globalPos].pos_col -= columns * PRECISION;
				}
				int posicion = (cellsEntrada[globalPos].pos_row / PRECISION) * columns + (cellsEntrada[globalPos].pos_col / PRECISION);
				atomicAdd(&arrayCulCells[posicion],1);
			}
		}
	}
	reductionMax2(cellsEntrada, numCells[0], &s[0].history_max_age);
}

__global__ void nacimientoCelulas(Cell* cellsEntrada, int* arrayCulCells, int* arrayCul, Cell* newCellsDevice, int* aliveDevice, int* stepNewCellsDevice, 
	int* banderaDevice, int* numCells, int columns, Statistics* stats)
{
	int pos = 0;
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalPos < numCells[0]){
		if(cellsEntrada[globalPos].alive){
			int posicion = (cellsEntrada[globalPos].pos_row / PRECISION) * columns + (cellsEntrada[globalPos].pos_col / PRECISION);
			int food = arrayCul[posicion];
			int count = arrayCulCells[posicion];
			int my_food = food / count;
			cellsEntrada[globalPos].storage += my_food;

			if ( cellsEntrada[globalPos].age > 30 && cellsEntrada[globalPos].storage > ENERGY_NEEDED_TO_SPLIT ) {
				atomicAdd(aliveDevice,1);
				pos = atomicAdd(stepNewCellsDevice,1);	
				atomicAdd(&stats[0].history_total_cells,1);		
				newCellsDevice[ pos ] = cellsEntrada[globalPos];

					// Split energy stored and update age in both cells
				cellsEntrada[globalPos].storage /= 2;
				newCellsDevice[ pos ].storage /= 2;
				cellsEntrada[globalPos].age = 1;
				newCellsDevice[ pos ].age = 1;

					// Random seed for the new cell, obtained using the parent random sequence
				newCellsDevice[ pos ].random_seq[0] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				newCellsDevice[ pos ].random_seq[1] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				newCellsDevice[ pos ].random_seq[2] = (unsigned short)glibc_nrand48( cellsEntrada[globalPos].random_seq );
				
				cell_new_direction( &cellsEntrada[globalPos] );
				cell_new_direction( &newCellsDevice[ pos ] );
				
				// Mutations of the movement genes in both cells
				cell_mutation( &cellsEntrada[globalPos] );
				cell_mutation( &newCellsDevice[ pos ] );
			}
		}
	} // End cell actions
}


/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */


#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, int *culture, int num_cells, Cell *cells, int num_cells_alive, Statistics sim_stat ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;

	printf("Iteration: %d\n", iteration );
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( culture, i, j ) >= 20 * PRECISION ) symbol = '+';
			else if ( accessMat( culture, i, j ) >= 10 * PRECISION ) symbol = '*';
			else if ( accessMat( culture, i, j ) >= 5 * PRECISION ) symbol = '.';
			else symbol = ' ';

			int t;
			int counter = 0;
			for( t=0; t<num_cells; t++ ) {
				int row = (int)(cells[t].pos_row / PRECISION);
				int col = (int)(cells[t].pos_col / PRECISION);
				if ( cells[t].alive && row == i && col == j ) {
					counter ++;
				}
			}
			if ( counter > 9 ) printf("(M)" );
			else if ( counter > 0 ) printf("(%1d)", counter );
			else printf(" %c ", symbol );
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	printf("Num_cells_alive: %04d\nHistory( Cells: %04d, Dead: %04d, Max.alive: %04d, Max.new: %04d, Max.dead: %04d, Max.age: %04d, Max.food: %6f )\n\n", 
		num_cells_alive, 
		sim_stat.history_total_cells, 
		sim_stat.history_dead_cells, 
		sim_stat.history_max_alive_cells, 
		sim_stat.history_max_new_cells, 
		sim_stat.history_max_dead_cells, 
		sim_stat.history_max_age,
		(float)sim_stat.history_max_food / PRECISION
	);
}
#endif

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<rows> <columns> <maxIter> <max_food> <food_density> <food_level> <short_rnd1> <short_rnd2> <short_rnd3> <num_cells>\n");
	fprintf(stderr,"\tOptional arguments for special food spot: [ <row> <col> <size_rows> <size_cols> <density> <level> ]\n");
	fprintf(stderr,"\n");
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j;

	// Simulation data
	int max_iter;			// Maximum number of simulation steps
	int rows, columns;		// Cultivation area sizes
	int *culture;			// Cultivation area values
	int *culture_cells;		// Ancillary structure to count the number of cells in a culture space

	float max_food;			// Maximum level of food on any position
	float food_density;		// Number of food sources introduced per step
	float food_level;		// Maximum number of food level in a new source

	bool food_spot_active = false;	// Special food spot: Active
	int food_spot_row = 0;		// Special food spot: Initial row
	int food_spot_col = 0;		// Special food spot: Initial row
	int food_spot_size_rows = 0;	// Special food spot: Rows size
	int food_spot_size_cols = 0;	// Special food spot: Cols size
	float food_spot_density = 0.0f;	// Special food spot: Food density
	float food_spot_level = 0.0f;	// Special food spot: Food level

	unsigned short init_random_seq[3];	// Status of the init random sequence
	unsigned short food_random_seq[3];	// Status of the food random sequence
	unsigned short food_spot_random_seq[3];	// Status of the special food spot random sequence

	int	num_cells;		// Number of cells currently stored in the list
	Cell	*cells;			// List to store cells information

	// Statistics
	Statistics sim_stat;	
	sim_stat.history_total_cells = 0;
	sim_stat.history_dead_cells = 0;
	sim_stat.history_max_alive_cells = 0;
	sim_stat.history_max_new_cells = 0;
	sim_stat.history_max_dead_cells = 0;
	sim_stat.history_max_age = 0;
	sim_stat.history_max_food = 0.0f;

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 11) {
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read culture sizes, maximum number of iterations */
	rows = atoi( argv[1] );
	columns = atoi( argv[2] );
	max_iter = atoi( argv[3] );

	/* 1.3. Food data */
	max_food = atof( argv[4] );
	food_density = atof( argv[5] );
	food_level = atof( argv[6] );

	/* 1.4. Read random sequences initializer */
	for( i=0; i<3; i++ ) {
		init_random_seq[i] = (unsigned short)atoi( argv[7+i] );
	}

	/* 1.5. Read number of cells */
	num_cells = atoi( argv[10] );
	//num_cellsAux2 = num_cellsAux2*2;

	/* 1.6. Read special food spot */
	if (argc > 11 ) {
		if ( argc < 17 ) {
			fprintf(stderr, "-- Error in number of special-food-spot arguments in the command line\n\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		else {
			food_spot_active = true;
			food_spot_row = atoi( argv[11] );
			food_spot_col = atoi( argv[12] );
			food_spot_size_rows = atoi( argv[13] );
			food_spot_size_cols = atoi( argv[14] );
			food_spot_density = atof( argv[15] );
			food_spot_level = atof( argv[16] );

			// Check non-used trailing arguments
			if ( argc > 17 ) {
				fprintf(stderr, "-- Error: too many arguments in the command line\n\n");
				show_usage( argv[0] );
				exit( EXIT_FAILURE );
			}
		}
	}

#ifdef DEBUG
	/* 1.7. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
	printf("Arguments, Max.food: %f, Food density: %f, Food level: %f\n", max_food, food_density, food_level);
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", init_random_seq[0], init_random_seq[1], init_random_seq[2]);
	if ( food_spot_active ) {
		printf("Arguments, Food_spot, pos(%d,%d), size(%d,%d), Density: %f, Level: %f\n",
			food_spot_row, food_spot_col, food_spot_size_rows, food_spot_size_cols, food_spot_density, food_spot_level );
	}
	printf("Initial cells: %d\n", num_cells );
#endif // DEBUG


	/* 1.8. Initialize random sequences for food dropping */
	for( i=0; i<3; i++ ) {
		food_random_seq[i] = (unsigned short)glibc_nrand48( init_random_seq );
		food_spot_random_seq[i] = (unsigned short)glibc_nrand48( init_random_seq );
	}

	/* 1.9. Initialize random sequences of cells */
	cells = (Cell *)malloc( sizeof(Cell) * (size_t)num_cells );
	if ( cells == NULL ) {
		fprintf(stderr,"-- Error allocating: %d cells\n", num_cells );
		exit( EXIT_FAILURE );
	}
	for( i=0; i<num_cells; i++ ) {
		// Initialize the cell ramdom sequences
		for( j=0; j<3; j++ ) 
			cells[i].random_seq[j] = (unsigned short)glibc_nrand48( init_random_seq );
	}


#ifdef DEBUG
	/* 1.10. Print random seed of the initial cells */
#endif // DEBUG


	// CUDA start
	cudaSetDevice(0);
	cudaDeviceSynchronize();

	/* 2. Start global timer */
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

#include "cuda_check.h"

	int num_cellsAux2 = 2000000;
	if(num_cells > num_cellsAux2){
		num_cellsAux2 = num_cells;
	}
	num_cellsAux2 = num_cellsAux2 * 10;
	Cell* cells2 = (Cell *)malloc( sizeof(Cell) * (size_t)num_cells );
	int tamanoBloq=1024; 
	int numeroBloq;
	int numeroBloq2=0;
	int numeroBloq3=0;
	int numeroBloq4=0;
	numeroBloq=(rows*columns)/tamanoBloq;
    if ((rows*columns)%tamanoBloq!=0) 
    {
        numeroBloq++;
    }
    numeroBloq2=num_cells*2/tamanoBloq;
    if (num_cells%tamanoBloq!=0) 
    {
        numeroBloq2++;
    }

	/* 3. Initialize culture surface and initial cells */
	int *numAlive;
	int *numeroCelulas;
	int *bandera;
	Statistics *stats;

	numeroCelulas = (int *)malloc( sizeof(int) );
	numAlive = (int *)malloc( sizeof(int) );
	bandera = (int *)malloc( sizeof(int) );
	culture = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );
	culture_cells = (int *)malloc( sizeof(int) * (size_t)rows * (size_t)columns );


	stats = (Statistics *)malloc( sizeof(Statistics) * (size_t)1 * (size_t)1 );

	if ( culture == NULL || culture_cells == NULL ) {
		fprintf(stderr,"-- Error allocating culture structures for size: %d x %d \n", rows, columns );
		exit( EXIT_FAILURE );
	}
	int* pDevice;
	int* p2Device;
	int* cDevice;
	Cell* cellsDevice;
	Cell* newCellsDevice;
	Cell* cellsDevice2;
	int* aDevice;
	int* numeroCelulasDevice;
	int* numAliveDevice;
	int* stepDeadCellsDevice;
	int* stepNewCellsDevice;
	int* banderaDevice;
	int* numNewSourcesDevice;
	int* numNewSourcesDevice2;
	int* numNewSourcesDevice4;
	int* numNewSourcesDevice5;

	Statistics* statsDevice;

	int tamComida2 = (int)(food_spot_size_rows * food_spot_size_cols * food_spot_density);
	int tamComida = (int)(rows * columns * food_density);

	cudaMalloc((void **) &numAliveDevice, sizeof(int)); 
	cudaMalloc((void **) &numeroCelulasDevice, sizeof(int)); 
	cudaMalloc((void **) &stepDeadCellsDevice, sizeof(int)); 
	cudaMalloc((void **) &stepNewCellsDevice, sizeof(int)); 
	cudaMalloc((void **) &pDevice, sizeof(int) * (size_t)rows * (size_t)columns); 
	cudaMalloc((void **) &p2Device, sizeof(int) * (size_t)rows * (size_t)columns);
	cudaMalloc((void **) &cDevice, sizeof(int)); 
	cudaMalloc((void **) &aDevice, sizeof(int)); 
	cudaMalloc((void **) &cellsDevice, sizeof(Cell) * (size_t)num_cellsAux2); 
	cudaMalloc((void **) &statsDevice, sizeof(Statistics)); 
	cudaMalloc((void **) &cellsDevice2, sizeof(Cell) * (size_t)num_cellsAux2); 	
	cudaMalloc((void **) &newCellsDevice, sizeof(Cell) * (size_t)num_cellsAux2); 

	cudaMalloc((void **) &numNewSourcesDevice, sizeof(int) * (size_t)tamComida); 
	cudaMalloc((void **) &numNewSourcesDevice2, sizeof(int) * (size_t)tamComida); 
	cudaMalloc((void **) &numNewSourcesDevice4, sizeof(int) * (size_t)tamComida2); 
	cudaMalloc((void **) &numNewSourcesDevice5, sizeof(int) * (size_t)tamComida2); 

	stats[0] = sim_stat;

	cudaMemset ( p2Device, 0,  sizeof(int) * (size_t)rows * (size_t)columns);

	cudaMemset ( pDevice, 0,  sizeof(int) * (size_t)rows * (size_t)columns);

	cudaMemcpy( cellsDevice, cells, sizeof(Cell) * num_cells ,cudaMemcpyHostToDevice );

	inicializarCelulas<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, num_cells, rows, columns);

	// Statistics: Initialize total number of cells, and max. alive
	sim_stat.history_total_cells = num_cells;
	sim_stat.history_max_alive_cells = num_cells;

#ifdef DEBUG
	/* Show initial cells data */
	printf("Initial cells data: %d\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("\tCell %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i, 
				(float)cells[i].pos_row / PRECISION, 
				(float)cells[i].pos_col / PRECISION, 
				(float)cells[i].mov_row / PRECISION, 
				(float)cells[i].mov_col / PRECISION, 
				(float)cells[i].choose_mov[0] / PRECISION, 
				(float)cells[i].choose_mov[1] / PRECISION, 
				(float)cells[i].choose_mov[2] / PRECISION, 
				(float)cells[i].storage / PRECISION,
				cells[i].age );
	}
#endif // DEBUG

	/* 4. Simulation */
	int current_max_food = 0;
	int num_cells_alive = num_cells;
	int iter;
	int max_food_int = max_food * PRECISION;
	numeroCelulas[0] = num_cells;
	numAlive[0] = num_cells_alive;
	stats[0].history_total_cells = num_cells;

	int row, col, food;
	int numeroBloq5;
	numeroBloq5=num_cells/tamanoBloq;
	if(num_cells%tamanoBloq!=0){
		numeroBloq5++;
	}
	numeroBloq3=tamComida/tamanoBloq;
    if (tamComida%tamanoBloq!=0) 
    {
        numeroBloq3++;
    }

    numeroBloq4=tamComida2/tamanoBloq;
    if (tamComida2%tamanoBloq!=0) 
    {
        numeroBloq4++;
    }

    int num_new_sources;
    int num_new_soruces2;

	num_new_sources = (int)(rows * columns * food_density);

	int* numNewSources5;
	int* numNewSources4;
	int* numNewSources = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
	int* numNewSources2 = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
	for (i=0; i<num_new_sources; i++) {
		row = int_urand48( rows, food_random_seq );
		col = int_urand48( columns, food_random_seq );
		food = int_urand48( food_level * PRECISION, food_random_seq );
		numNewSources[i] = (int)(row) * columns + (int)(col);
		numNewSources2[i] = food;
	}

	cudaMemcpy( numNewSourcesDevice, numNewSources, sizeof(int) * num_new_sources ,cudaMemcpyHostToDevice );
	cudaMemcpy( numNewSourcesDevice2, numNewSources2, sizeof(int) * num_new_sources ,cudaMemcpyHostToDevice );

	repartoComidaFoodSpot<<<numeroBloq3, tamanoBloq, sizeof(int)*tamanoBloq>>>(numNewSourcesDevice, numNewSourcesDevice2, num_new_sources, columns, pDevice);

	if ( food_spot_active ) {

		num_new_soruces2 = (int)(food_spot_size_rows * food_spot_size_cols * food_spot_density);
		numNewSources4 = (int *)malloc( sizeof(int) * (size_t)num_new_soruces2);
		numNewSources5 = (int *)malloc( sizeof(int) * (size_t)num_new_soruces2);
		//int* numNewSources6 = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
		for (i=0; i<num_new_soruces2; i++) {
			row = food_spot_row + int_urand48( food_spot_size_rows, food_spot_random_seq );
			col = food_spot_col + int_urand48( food_spot_size_cols, food_spot_random_seq );
			food = int_urand48( food_spot_level * PRECISION, food_spot_random_seq );
			numNewSources4[i] = (int)(row) * columns + (int)(col);
			numNewSources5[i] = food;
		}
		cudaMemcpy( numNewSourcesDevice4, numNewSources4, sizeof(int) * num_new_soruces2 ,cudaMemcpyHostToDevice );
		cudaMemcpy( numNewSourcesDevice5, numNewSources5, sizeof(int) * num_new_soruces2 ,cudaMemcpyHostToDevice );
		repartoComidaFoodSpot<<<numeroBloq4, tamanoBloq, sizeof(int)*tamanoBloq>>>(numNewSourcesDevice4, numNewSourcesDevice5, num_new_soruces2, columns, pDevice);
	}

	cudaMemcpy( statsDevice, stats, sizeof(Statistics)  ,cudaMemcpyHostToDevice );

	cudaMemcpy( numeroCelulasDevice, numeroCelulas, sizeof(int) ,cudaMemcpyHostToDevice );

	cudaMemcpy( numAliveDevice, numAlive, sizeof(int) ,cudaMemcpyHostToDevice );

	cudaMemset ( cellsDevice2, 0,  sizeof(Cell) ) ;

	cudaMemset ( stepDeadCellsDevice, 0,  sizeof(int) ) ;

	cudaMemset ( stepDeadCellsDevice, 0,  sizeof(int) ) ;

	/*
	Cell *new_cells2 = (Cell *)malloc( sizeof(Cell) * num_cells );
	if ( new_cells2 == NULL ) {
		fprintf(stderr,"-- Error allocating new cells structures for: %d cells\n", num_cells );
		exit( EXIT_FAILURE );
	}*/

	cudaMemset( newCellsDevice, 0,  sizeof(Cell) ) ;

	//cudaMemcpy( newCellsDevice, new_cells2, sizeof(Cell) ,cudaMemcpyHostToDevice);

	cudaMemset ( stepNewCellsDevice, 0,  sizeof(int) ) ;
	//cudaMemcpy( stepNewCellsDevice, stepNewCells, sizeof(int)  ,cudaMemcpyHostToDevice );

	//cudaMemcpy( aDevice, aliveList, sizeof(int)  ,cudaMemcpyHostToDevice );

	cudaMemset ( aDevice, 0,  sizeof(int) ) ;

	cudaMemset ( cDevice, 0,  sizeof(int) ) ;

	//cudaMemcpy( cDevice, maxFoodTemp, sizeof(int)  ,cudaMemcpyHostToDevice );

	//int numeroBloq3;


for( iter=0; iter<max_iter && current_max_food <= max_food_int && num_cells_alive > 0; iter++ ) {
		

		movimientoCelulas<<<numeroBloq5, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, rows, columns, numAliveDevice, stepDeadCellsDevice, statsDevice, numeroCelulasDevice, pDevice, p2Device/*, foodShareDevice*/);


		nacimientoCelulas<<<numeroBloq5, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, p2Device, pDevice, newCellsDevice, numAliveDevice,
			stepNewCellsDevice, banderaDevice, numeroCelulasDevice, columns, statsDevice);


		limpiarCelulas<<<numeroBloq5, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice, numeroCelulasDevice, aDevice, cellsDevice2, rows, columns, pDevice);
				num_new_sources = (int)(rows * columns * food_density) ;


		/*
		int* numNewSources = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
		int* numNewSources2 = (int *)malloc( sizeof(int) * (size_t)num_new_sources);

		//int* numNewSources3 = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
		for (i=0; i<num_new_sources * 1/2; i++) {
			row = int_urand48( rows, food_random_seq );
			col = int_urand48( columns, food_random_seq );
			food = int_urand48( food_level * PRECISION, food_random_seq );
			numNewSources[i] = (int)(row) * columns + (int)(col);
			numNewSources2[i] = food;
		}

		if ( food_spot_active ) {	
			numNewSources4 = (int *)malloc( sizeof(int) * (size_t)num_new_soruces2);
			numNewSources5 = (int *)malloc( sizeof(int) * (size_t)num_new_soruces2);			
			for (i=0; i<num_new_soruces2 * 1/2; i++) {
				row = food_spot_row + int_urand48( food_spot_size_rows, food_spot_random_seq );
				col = food_spot_col + int_urand48( food_spot_size_cols, food_spot_random_seq );
				food = int_urand48( food_spot_level * PRECISION, food_spot_random_seq );
				numNewSources4[i] = (int)(row) * columns + (int)(col);
				numNewSources5[i] = food;
			}
		}*/

		/*
		cudaMemcpy( bandera, aDevice, sizeof(int) ,cudaMemcpyDeviceToHost );

		numeroBloq2 = bandera[0]/tamanoBloq;
		if (bandera[0]%tamanoBloq!=0) 
    	{
        	numeroBloq2++;
    	}*/

		limpiarCelulas2<<<numeroBloq5, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice2, aDevice, cellsDevice, numeroCelulasDevice);

		/*
		for (i=num_new_sources * 1/2; i<num_new_sources * 3/4; i++) {
			row = int_urand48( rows, food_random_seq );
			col = int_urand48( columns, food_random_seq );
			food = int_urand48( food_level * PRECISION, food_random_seq );
			numNewSources[i] = (int)(row) * columns + (int)(col);
			numNewSources2[i] = food;
		}

		if ( food_spot_active ) {
			for (i=num_new_soruces2 * 1/2; i<num_new_soruces2 * 3/4; i++) {
				row = food_spot_row + int_urand48( food_spot_size_rows, food_spot_random_seq );
				col = food_spot_col + int_urand48( food_spot_size_cols, food_spot_random_seq );
				food = int_urand48( food_spot_level * PRECISION, food_spot_random_seq );
				numNewSources4[i] = (int)(row) * columns + (int)(col);
				numNewSources5[i] = food;
			}

		}*/
		/*
		cudaMemcpy( bandera2, stepNewCellsDevice, sizeof(int)  ,cudaMemcpyDeviceToHost );

		numeroBloq2 = bandera2[0]/tamanoBloq;
		if (bandera2[0]%tamanoBloq!=0)
		{
			numeroBloq2++;
		}*/	
		
		anyadirCelulas<<<numeroBloq5, tamanoBloq, sizeof(int)*tamanoBloq>>>(stepNewCellsDevice, newCellsDevice, cellsDevice, numeroCelulasDevice);
		
		/*
		bandera[0]+=bandera2[0];
		numeroBloq2 = bandera[0]/tamanoBloq;
		if (bandera[0]%tamanoBloq!=0) 
    	{
        	numeroBloq2++;
    	}*/

		//anyadirCelulas2<<<numeroBloq2, tamanoBloq, sizeof(int)*tamanoBloq>>>(cellsDevice2, stepNewCellsDevice, numeroCelulasDevice, cellsDevice); 

		reductionMax<<<numeroBloq, tamanoBloq, sizeof(int)*tamanoBloq>>>(pDevice, rows*columns, cDevice, p2Device, numeroCelulasDevice, stepNewCellsDevice, aDevice, statsDevice, stepDeadCellsDevice, numAliveDevice);

		final<<<1, 1, sizeof(int)*1>>>(cDevice, statsDevice);

		if(iter!=max_iter-1){
			int* numNewSources = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
			int* numNewSources2 = (int *)malloc( sizeof(int) * (size_t)num_new_sources);
			for (i=0; i<num_new_sources; i++) {
				row = int_urand48( rows, food_random_seq );
				col = int_urand48( columns, food_random_seq );
				food = int_urand48( food_level * PRECISION, food_random_seq );
				numNewSources[i] = (int)(row) * columns + (int)(col);
				numNewSources2[i] = food;
			}

			cudaMemcpy( numNewSourcesDevice, numNewSources, sizeof(int) * num_new_sources ,cudaMemcpyHostToDevice );
			cudaMemcpy( numNewSourcesDevice2, numNewSources2, sizeof(int) * num_new_sources ,cudaMemcpyHostToDevice );

			repartoComidaFoodSpot<<<numeroBloq3, tamanoBloq, sizeof(int)*tamanoBloq>>>(numNewSourcesDevice, numNewSourcesDevice2, num_new_sources, columns, pDevice);

			if ( food_spot_active ) {
				for (i=0; i<num_new_soruces2; i++) {
					row = food_spot_row + int_urand48( food_spot_size_rows, food_spot_random_seq );
					col = food_spot_col + int_urand48( food_spot_size_cols, food_spot_random_seq );
					food = int_urand48( food_spot_level * PRECISION, food_spot_random_seq );
					numNewSources4[i] = (int)(row) * columns + (int)(col);
					numNewSources5[i] = food;
				}
				cudaMemcpy( numNewSourcesDevice4, numNewSources4, sizeof(int) * num_new_soruces2 ,cudaMemcpyHostToDevice );
				cudaMemcpy( numNewSourcesDevice5, numNewSources5, sizeof(int) * num_new_soruces2 ,cudaMemcpyHostToDevice );
				repartoComidaFoodSpot<<<numeroBloq4, tamanoBloq, sizeof(int)*tamanoBloq>>>(numNewSourcesDevice4, numNewSourcesDevice5, num_new_soruces2, columns, pDevice);
			}
		}

    	Statistics prev_stats = stats[0];

		cudaMemcpy( stats, statsDevice, sizeof(Statistics)  ,cudaMemcpyDeviceToHost );

    	if (iter > 0){
            num_cells_alive += (stats[0].history_total_cells - prev_stats.history_total_cells) - (stats[0].history_dead_cells - prev_stats.history_dead_cells);
    	}

        if (iter == 0){
        	cudaMemcpy( bandera, numeroCelulasDevice, sizeof(int) ,cudaMemcpyDeviceToHost );
        	num_cells_alive = bandera[0];
        }
		current_max_food = stats[0].history_max_food;

		numeroBloq2 = num_cells_alive/tamanoBloq;
		if (num_cells_alive%tamanoBloq!=0) 
    	{
        	numeroBloq2++;
    	}

    	numeroBloq5 = num_cells_alive/tamanoBloq;
		if (num_cells_alive%tamanoBloq!=0) 
    	{
        	numeroBloq5++;
    	}


#ifdef DEBUG
		/* 4.10. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, culture, num_cells, cells, num_cells_alive, sim_stat );
#endif // DEBUG
	}

	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	// CUDA stop
	cudaDeviceSynchronize();

	/* 5. Stop global time */
	ttotal = cp_Wtime() - ttotal;

#ifdef DEBUG
	printf("List of cells at the end of the simulation: %d\n\n", num_cells );
	for( i=0; i<num_cells; i++ ) {
		printf("Cell %d, Alive: %d, Pos(%f,%f), Mov(%f,%f), Choose_mov(%f,%f,%f), Storage: %f, Age: %d\n",
				i,
				cells[i].alive,
				(float)cells[i].pos_row / PRECISION, 
				(float)cells[i].pos_col / PRECISION, 
				(float)cells[i].mov_row / PRECISION, 
				(float)cells[i].mov_col / PRECISION, 
				(float)cells[i].choose_mov[0] / PRECISION, 
				(float)cells[i].choose_mov[1] / PRECISION, 
				(float)cells[i].choose_mov[2] / PRECISION, 
				(float)cells[i].storage / PRECISION,
				cells[i].age );
	}
#endif // DEBUG

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	//printf("Tiempo del bucle de inicializacion: %lf\n", totalSuma2);

	//printf("Tiempo del bucle de movimiento: %lf\n", totalSuma3);

	//printf("Tiempo del bucle de nacimiento: %lf\n", totalSuma4);

	//printf("Tiempo del bucle de 4.8: %lf\n", totalSuma5);

	//printf("Tiempo del bucle de 4.1: %lf\n", totalSuma7);

	//printf("Secuencia principal: %lf\n", totalSuma6);

	/* 6.2. Results: Number of iterations and other statistics */
	cudaMemcpy( stats, statsDevice, sizeof(Statistics) * 1 * 1 ,cudaMemcpyDeviceToHost );
	sim_stat = stats[0];
	printf("Result: %d, ", iter);
	printf("%d, %d, %d, %d, %d, %d, %d, %f\n", 
		num_cells_alive, 
		sim_stat.history_total_cells, 
		sim_stat.history_dead_cells, 
		sim_stat.history_max_alive_cells, 
		sim_stat.history_max_new_cells, 
		sim_stat.history_max_dead_cells, 
		sim_stat.history_max_age,
		(float)sim_stat.history_max_food / PRECISION
	);

	/* 7. Free resources */	
	free( culture );
	free( culture_cells );
	free( cells );

	/* 8. End */
	return 0;
}