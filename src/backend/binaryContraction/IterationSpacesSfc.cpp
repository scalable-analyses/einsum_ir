#include "IterationSpacesSfc.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::IterationSpacesSfc::init( std::vector< dim_t >   const * i_loop_dim_type,
                                                   std::vector< exec_t >  const * i_loop_exec_type,
                                                   std::vector< int64_t > const * i_loop_sizes,
                                                   std::vector< int64_t > const * i_loop_strides_left,
                                                   std::vector< int64_t > const * i_loop_strides_right,
                                                   std::vector< int64_t > const * i_loop_strides_out_aux,
                                                   std::vector< int64_t > const * i_loop_strides_out,
                                                   int64_t                        i_num_threads){

  m_loop_dim_type        = i_loop_dim_type;
  m_loop_exec_type       = i_loop_exec_type;
  m_loop_sizes           = i_loop_sizes;

  m_loop_strides.resize(4);
  m_loop_strides[0] = i_loop_strides_left;
  m_loop_strides[1] = i_loop_strides_right;
  m_loop_strides[2] = i_loop_strides_out;
  m_loop_strides[3] = i_loop_strides_out_aux;

  m_num_threads = i_num_threads;
}

einsum_ir::err_t einsum_ir::backend::IterationSpacesSfc::compile(){
 err_t l_err = err_t::UNDEFINED_ERROR;

  //calculate number of generated tasks
  m_num_tasks = 1;
  m_num_parallel_loops = 0;
  for( size_t l_id = 0; l_id < m_loop_dim_type->size(); l_id++ ){
    if( m_loop_exec_type->at(l_id) == einsum_ir::OMP ||
        m_loop_exec_type->at(l_id) == einsum_ir::SFC    ){
      if( !m_num_parallel_loops ){
        m_parallel_loops.begin = l_id;
      }
      if( m_loop_dim_type->at(l_id) != einsum_ir::K ){
        m_num_tasks *= m_loop_sizes->at(l_id);
      }
      m_num_parallel_loops += 1;
    }
  }
  m_parallel_loops.end = m_parallel_loops.begin + m_num_parallel_loops;
  if( m_num_parallel_loops == 0 ){
    return err_t::SUCCESS;
  }

  //assigns parallel dimensions to three types omp, sfc_n, sfc_m
  //restrictions:
  //  all parallel dims must be consecutive
  //  first omp dims can be of type m,n or c
  //  second sfc dims of type m
  //  third sfc dims of type n
  // example: 
  //  dim_t : ...  c1,  m1,  n1,  m2,  n2,  n3,  k1,  k2, ...
  //  exec_t: ... omp, omp, omp, sfc, sfc, sfc, sfc, sfc, ...
  int64_t l_last_found_type = 0;
  for( int64_t l_id = m_parallel_loops.begin; l_id < m_parallel_loops.end ; l_id++ ){
    if( m_loop_exec_type->at(l_id) == einsum_ir::OMP &&
        l_last_found_type      <= 1 ){
      if( l_last_found_type == 0 ){
        m_omp_loops.begin = l_id;
      }
      m_omp_loops.end = l_id + 1;
      l_last_found_type = 1;
    }
    else if( m_loop_exec_type->at(l_id) == einsum_ir::SFC &&
             m_loop_dim_type->at(l_id)  == einsum_ir::M &&
             l_last_found_type <= 2){
      m_sfc_tasks_m *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 1 ){
        m_sfc_loops_m.begin = l_id;
      }
      m_sfc_loops_m.end = l_id + 1;
      l_last_found_type = 2;
    }
    else if( m_loop_exec_type->at(l_id) == einsum_ir::SFC &&
             m_loop_dim_type->at(l_id)  == einsum_ir::N   &&
             l_last_found_type <= 3 ){
      m_sfc_tasks_n *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 2 ){
        m_sfc_loops_n.begin = l_id;
      }
      m_sfc_loops_n.end = l_id + 1;
      l_last_found_type = 3;
    }
    else if( m_loop_exec_type->at(l_id) == einsum_ir::SFC &&
             m_loop_dim_type->at(l_id)  == einsum_ir::K ){
      m_sfc_tasks_k *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 4 ){
        m_sfc_loops_k.begin = l_id;
      }
      m_sfc_loops_k.end = l_id + 1;
      l_last_found_type = 4;
    }
    else{
      return err_t::COMPILATION_FAILED;
    }
  }
  std::cout << m_num_tasks << " " << m_sfc_tasks_m << " " << m_sfc_tasks_n << std::endl; 

  //convert strides to offsets
  int64_t l_num_tensors = m_loop_strides.size();
  m_movement_offsets.resize(l_num_tensors );
  for(int64_t l_io_tensor = 0; l_io_tensor < l_num_tensors ; l_io_tensor++){
    //TODO fix
    dim_t l_sfc_primary = m_sfc_tasks_m > m_sfc_tasks_n ? dim_t::M : dim_t::N;
    convertStridesToOffsets( l_sfc_primary,
                             *m_loop_strides.at(l_io_tensor),
                             m_movement_offsets[l_io_tensor] );
  } 


  //allocate memory for iteration space
  m_dim_movements.resize(     m_num_threads );
  m_initial_offsets.resize(   m_num_threads );
  m_thread_work_space.resize( m_num_threads );
  int64_t l_tasks_per_thread = m_num_tasks / m_num_threads + (m_num_tasks % m_num_threads != 0);
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_id * l_tasks_per_thread;
    int64_t l_end   = l_begin     + l_tasks_per_thread;
    l_begin = l_begin < m_num_tasks ? l_begin : m_num_tasks;
    l_end   = l_end   < m_num_tasks ? l_end   : m_num_tasks;
    l_begin *= m_sfc_tasks_k;
    l_end   *= m_sfc_tasks_k;

    m_thread_work_space[l_thread_id].begin = l_begin;
    m_thread_work_space[l_thread_id].end   = l_end;

    m_dim_movements[l_thread_id].resize( l_end - l_begin );
    m_initial_offsets[l_thread_id].resize( l_num_tensors );
  }

//create 1D Map of task
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = m_thread_work_space[l_thread_id].begin;
    int64_t l_end   = m_thread_work_space[l_thread_id].end;

    int64_t l_id_sfc_m_old, l_id_sfc_n_old, l_id_sfc_k_old, l_id_omp_old;
    SfcOracle3d(&l_id_sfc_m_old, &l_id_sfc_n_old, &l_id_sfc_k_old, &l_id_omp_old, l_begin, m_thread_work_space[l_thread_id] );
    //calculate initial thread offsets
    for(int64_t l_io_tensor = 0; l_io_tensor < l_num_tensors; l_io_tensor++){
      int64_t l_offset = calculateInitialOffset( l_id_omp_old,
                                                 l_id_sfc_m_old,
                                                 l_id_sfc_n_old,
                                                 0,
                                                 *m_loop_strides.at(l_io_tensor) );
      m_initial_offsets[l_thread_id][l_io_tensor] = l_offset;
    }

    //calculate movements
    for( int64_t l_id = l_begin; l_id < l_end; l_id++ ){
      int64_t l_id_sfc_m_new, l_id_sfc_n_new, l_id_sfc_k_new, l_id_omp_new; 
      SfcOracle3d(&l_id_sfc_m_new, &l_id_sfc_n_new, &l_id_sfc_k_new, &l_id_omp_new, l_id+1, m_thread_work_space[l_thread_id] );

      if( l_id_omp_new != l_id_omp_old ){
        uint8_t l_move = getMaxDimJump( m_omp_loops, l_id_omp_new, l_id_omp_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move; 
      }
      else if( l_id_sfc_m_new != l_id_sfc_m_old ){
        uint8_t l_move = getMaxDimJump( m_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }
      else if( l_id_sfc_n_new != l_id_sfc_n_old ){
        uint8_t l_move = getMaxDimJump( m_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }
      else if( l_id_sfc_k_new != l_id_sfc_k_old ){
        uint8_t l_move = getMaxDimJump( m_sfc_loops_k, l_id_sfc_k_new, l_id_sfc_k_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }
      l_id_sfc_m_old = l_id_sfc_m_new;
      l_id_sfc_n_old = l_id_sfc_n_new;
      l_id_sfc_k_old = l_id_sfc_k_new;
      l_id_omp_old = l_id_omp_new;
    }
  }

  return err_t::SUCCESS;
}

int64_t einsum_ir::backend::IterationSpacesSfc::calculateInitialOffset( int64_t i_id_omp,
                                                                        int64_t i_id_sfc_m,
                                                                        int64_t i_id_sfc_n,
                                                                        int64_t i_id_sfc_k,
                                                                        std::vector< int64_t > const & i_strides ) {

  int64_t l_offset = 0;
  for (int64_t l_id = m_sfc_loops_m.end - 1; l_id >= m_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_m % l_size) * l_stride;
    i_id_sfc_m /= l_size;
  }
  for (int64_t l_id = m_sfc_loops_n.end - 1; l_id >= m_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_n % l_size) * l_stride;
    i_id_sfc_n /= l_size;
  }
  for (int64_t l_id = m_sfc_loops_k.end - 1; l_id >= m_sfc_loops_k.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_sfc_k % l_size) * l_stride;
    i_id_sfc_k /= l_size;
  }
  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_omp % l_size) * l_stride;
    i_id_omp /= l_size;
  }

  return l_offset;
}




void einsum_ir::backend::IterationSpacesSfc::convertStridesToOffsets( dim_t   i_sfc_primary,
                                                                      std::vector< int64_t > const & i_strides,
                                                                      std::vector< int64_t >       & io_offsets ) {
  // id 0 is always set to 0
  // last id can be used for diagonal jumps
  io_offsets.resize( m_num_parallel_loops + 1); 
  int64_t l_first = m_parallel_loops.begin - 1;

  int64_t l_all_offsets_sfc_m = 0;
  for (int64_t l_id = m_sfc_loops_m.end - 1; l_id >= m_sfc_loops_m.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id - l_first] = l_stride - l_all_offsets_sfc_m;
    l_all_offsets_sfc_m += (l_size - 1) * l_stride;
  }
  
  int64_t l_all_offsets_sfc_n = 0;
  for (int64_t l_id = m_sfc_loops_n.end - 1; l_id >= m_sfc_loops_n.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id - l_first] = l_stride - l_all_offsets_sfc_n;
    l_all_offsets_sfc_n += (l_size - 1) * l_stride;
  }

  int64_t l_all_offsets_sfc_k = 0;
  for (int64_t l_id = m_sfc_loops_k.end - 1; l_id >= m_sfc_loops_k.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id - l_first] = l_stride - l_all_offsets_sfc_k;
    l_all_offsets_sfc_k += (l_size - 1) * l_stride;
  }

  int64_t l_id_sfc_m, l_id_sfc_n, l_id_omp;
  SfcOracle2d(&l_id_sfc_m, &l_id_sfc_n, &l_id_omp, m_sfc_tasks_m*m_sfc_tasks_n-1);
  int64_t l_all_offsets_omp = calculateInitialOffset( l_id_omp,
                                                      l_id_sfc_m,
                                                      l_id_sfc_n,
                                                      0,
                                                      i_strides );

  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id - l_first] = l_stride - l_all_offsets_omp;
    l_all_offsets_omp += (l_size - 1) * l_stride;
  }
}

uint8_t einsum_ir::backend::IterationSpacesSfc::getMaxDimJump( range_t i_dim_loops,
                                                               int64_t i_id_new,
                                                               int64_t i_id_old ){

  int64_t l_dif = i_id_new - i_id_old;
  int64_t l_max_id = i_id_new > i_id_old ? i_id_new : i_id_old;
  for( int64_t l_di = i_dim_loops.end-1; l_di >= i_dim_loops.begin; l_di-- ){
    int64_t l_size = m_loop_sizes->at(l_di);
    if(l_max_id % l_size != 0){
      return (l_di - m_parallel_loops.begin + 1) * l_dif;
    }
    else{
      l_max_id /= l_size;
    }
  }

  return 0;
}

int64_t einsum_ir::backend::IterationSpacesSfc::getNumTasks( int64_t i_thread_id ){
  return m_dim_movements[i_thread_id].size();
}

void einsum_ir::backend::IterationSpacesSfc::addMovementOffsets( int64_t           i_thread_id, 
                                                                 int64_t           i_task_id,
                                                                 char    const ** io_offset_left,
                                                                 char    const ** io_offset_right,
                                                                 char          ** io_offset_out){
  uint8_t l_move =  m_dim_movements[i_thread_id][i_task_id];
  int64_t l_direction = l_move >> 7 ? -1 : 1;
  l_move = l_direction == 1 ? l_move : 256-l_move;
  //std::cout << l_move * l_direction << std::endl;
  //TODO could use c arrays instead 
  *io_offset_left  += l_direction * m_movement_offsets[0][l_move];
  *io_offset_right += l_direction * m_movement_offsets[1][l_move];
  *io_offset_out   += l_direction * m_movement_offsets[2][l_move];
}

void einsum_ir::backend::IterationSpacesSfc::addInitialOffsets( int64_t    i_thread_id,
                                                                char    const ** io_offset_left,
                                                                char    const ** io_offset_right,
                                                                char          ** io_offset_out) {
  *io_offset_left  += m_initial_offsets[i_thread_id][0];
  *io_offset_right += m_initial_offsets[i_thread_id][1];
  *io_offset_out   += m_initial_offsets[i_thread_id][2];
}

void einsum_ir::backend::IterationSpacesSfc::SfcOracle3d( int64_t *i_m, 
                                                          int64_t *i_n, 
                                                          int64_t *i_k,
                                                          int64_t *i_omp, 
                                                          int64_t  i_idx,
                                                          range_t  i_thread_range ){
  
  int l_idx_sfc2 = i_idx;
  if( m_sfc_tasks_k > 1 ){
    int l_w = m_sfc_tasks_k;
    int l_offset = i_thread_range.begin;
    int l_h = ( i_thread_range.end - l_offset ) / l_w;
    int l_idx_k;
    gilbert_d2xy(&l_idx_k, &l_idx_sfc2, i_idx-l_offset, l_w, l_h);
    *i_k = l_idx_k;
    l_idx_sfc2 += l_offset / l_w;
  }
  SfcOracle2d(i_m, i_n, i_omp, l_idx_sfc2);
}

void einsum_ir::backend::IterationSpacesSfc::SfcOracle2d( int64_t *i_m, 
                                                          int64_t *i_n,
                                                          int64_t *i_omp, 
                                                          int64_t  i_idx ){
  
  int l_w = m_sfc_tasks_m;
  int l_h = m_sfc_tasks_n;
  *i_omp = i_idx / (l_w*l_h);
  i_idx = i_idx % (l_w*l_h);

  int l_idx_m, l_idx_n;
  gilbert_d2xy(&l_idx_m, &l_idx_n, i_idx, l_w, l_h);

  *i_m = l_idx_m;
  *i_n = l_idx_n;
}

#define SIGN(A) (0 < (A) ? (1) : ( 0 == (A) ? (0) : (-1)))

int gilbert_d2xy_r( int dst_idx, int cur_idx,
                    int *xres, int *yres,
                    int ax,int ay,
                    int bx,int by ) {
  int nxt_idx;
  int w, h, x, y,
      dax, day,
      dbx, dby,
      di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = abs(ax + ay);
  h = abs(bx + by);

  x = *xres;
  y = *yres;

  // unit major direction
  dax = SIGN(ax);
  day = SIGN(ay);

  // unit orthogonal direction
  dbx = SIGN(bx);
  dby = SIGN(by);

  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax*di;
    *yres = y + day*di;
    return 0;
  }

  if (w == 1) {
    *xres = x + dbx*di;
    *yres = y + dby*di;
    return 0;
  }

  // floor function
  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;

  w2 = abs(ax2 + ay2);
  h2 = abs(bx2 + by2);

  if ((2*w) > (3*h)) {
    if ((w2 & 1) && (w > 2)) {
      // prefer even steps
      ax2 += dax;
      ay2 += day;
    }

    // long case: split in two parts only
    nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return gilbert_d2xy_r(dst_idx, cur_idx,  xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;

    *xres = x + ax2;
    *yres = y + ay2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax-ax2, ay-ay2, bx, by);
  }

  if ((h2 & 1) && (h > 2)) {
    // prefer even steps
    bx2 += dbx;
    by2 += dby;
  }

  // standard case: one step up, one long horizontal, one step down
  nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, bx2,by2, ax2,ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, ax,ay, bx-bx2,by-by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return gilbert_d2xy_r(dst_idx, cur_idx,
                        xres,yres,
                        -bx2, -by2,
                        -(ax-ax2), -(ay-ay2));
}

int einsum_ir::backend::IterationSpacesSfc::gilbert_d2xy( int *x, 
                                                          int *y, 
                                                          int  idx,
                                                          int  w,
                                                          int  h  ) {  
  *x = 0;
  *y = 0;

  //variables to indicate if movement through dimension possible without jump
  bool move_w_possible = w % 2 == 0 || h % 2 == 1;
  bool move_h_possible = w % 2 == 1 || h % 2 == 0;

  if ( (w >= h && move_w_possible) || !move_h_possible ) {
    return gilbert_d2xy_r(idx,0, x,y, w,0, 0,h);
  }
  return gilbert_d2xy_r(idx,0, x,y, 0,h, w,0);
}


int64_t einsum_ir::backend::IterationSpacesSfc::num_tasks( ){
  return m_num_tasks;
}