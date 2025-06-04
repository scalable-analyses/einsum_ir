#include "IterationSpace.h"
#include "../third_party/gilbertSFC.cpp"

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::binary::IterationSpace::init( std::vector< dim_t >   const * i_loop_dim_type,
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

einsum_ir::err_t einsum_ir::binary::IterationSpace::compile(){
 err_t l_err = err_t::UNDEFINED_ERROR;

  //calculate number of generated tasks
  int64_t l_num_tasks = 1;
  int64_t l_num_parallel_loops = 0;
  for( size_t l_id = 0; l_id < m_loop_dim_type->size(); l_id++ ){
    if( m_loop_exec_type->at(l_id) == exec_t::OMP ||
        m_loop_exec_type->at(l_id) == exec_t::SFC    ){
      if( !l_num_parallel_loops ){
        m_parallel_loops.begin = l_id;
      }
      if( m_loop_dim_type->at(l_id) != dim_t::K ){
        l_num_tasks *= m_loop_sizes->at(l_id);
      }
      l_num_parallel_loops += 1;
    }
  }
  m_parallel_loops.end = m_parallel_loops.begin + l_num_parallel_loops;
  if( l_num_parallel_loops == 0 ){
    return err_t::SUCCESS;
  }

  //assigns parallel dimensions to three types omp, sfc_n, sfc_m
  m_sfc_tasks_m = 1;
  m_sfc_tasks_n = 1;
  int64_t l_last_found_type = 0;
  for( int64_t l_id = m_parallel_loops.begin; l_id < m_parallel_loops.end ; l_id++ ){
    if( m_loop_exec_type->at(l_id) == exec_t::OMP &&
        l_last_found_type <= 1 ){
      if( l_last_found_type == 0 ){
        m_omp_loops.begin = l_id;
      }
      m_omp_loops.end = l_id + 1;
      l_last_found_type = 1;
    }
    else if( m_loop_exec_type->at(l_id) == exec_t::SFC &&
             m_loop_dim_type->at(l_id)  == dim_t::M &&
             l_last_found_type <= 2){
      m_sfc_tasks_m *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 1 ){
        m_sfc_loops_m.begin = l_id;
      }
      m_sfc_loops_m.end = l_id + 1;
      l_last_found_type = 2;
    }
    else if( m_loop_exec_type->at(l_id) == exec_t::SFC &&
             m_loop_dim_type->at(l_id)  == dim_t::N   &&
             l_last_found_type <= 3 ){
      m_sfc_tasks_n *= m_loop_sizes->at(l_id);
      if( l_last_found_type <= 2 ){
        m_sfc_loops_n.begin = l_id;
      }
      m_sfc_loops_n.end = l_id + 1;
      l_last_found_type = 3;
    }
    else{
      return err_t::COMPILATION_FAILED;
    }
  }

  //convert strides to offsets
  int64_t l_num_tensors = m_loop_strides.size();
  m_movement_offsets.resize(l_num_tensors );
  for(int64_t l_io_tensor = 0; l_io_tensor < l_num_tensors ; l_io_tensor++){
    m_movement_offsets[l_io_tensor].resize( l_num_parallel_loops );
    convert_strides_to_offsets( *m_loop_strides.at(l_io_tensor),
                                m_movement_offsets[l_io_tensor] );
  } 


  //allocate memory for iteration space
  std::vector< range_t > l_thread_work_space;
  m_dim_movements.resize(     m_num_threads );
  m_initial_offsets.resize(   m_num_threads );
  l_thread_work_space.resize( m_num_threads );
  int64_t l_tasks_per_thread = l_num_tasks / m_num_threads + (l_num_tasks % m_num_threads != 0);
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_id * l_tasks_per_thread;
    int64_t l_end   = l_begin     + l_tasks_per_thread;
    l_begin = l_begin < l_num_tasks ? l_begin : l_num_tasks;
    l_end   = l_end   < l_num_tasks ? l_end   : l_num_tasks;

    l_thread_work_space[l_thread_id].begin = l_begin;
    l_thread_work_space[l_thread_id].end   = l_end;

    m_dim_movements[l_thread_id].resize( l_end - l_begin );
    m_initial_offsets[l_thread_id].resize( l_num_tensors );
  }

//create 1D Map of task
#ifdef _OPENMP
#pragma omp parallel for num_threads(m_num_threads)
#endif
  for( int64_t l_thread_id = 0; l_thread_id < m_num_threads; l_thread_id++ ){
    int64_t l_begin = l_thread_work_space[l_thread_id].begin;
    int64_t l_end   = l_thread_work_space[l_thread_id].end;

    int64_t l_id_sfc_m_old, l_id_sfc_n_old, l_id_omp_old;
    sfc_oracle_2d( &l_id_sfc_m_old, &l_id_sfc_n_old, &l_id_omp_old, l_begin );

    //calculate initial thread offsets
    for(int64_t l_io_tensor = 0; l_io_tensor < l_num_tensors; l_io_tensor++){
      int64_t l_offset = calculate_offset( l_id_omp_old,
                                           l_id_sfc_m_old,
                                           l_id_sfc_n_old,
                                           *m_loop_strides.at(l_io_tensor) );
      m_initial_offsets[l_thread_id][l_io_tensor] = l_offset;
    }

    //calculate movements
    for( int64_t l_id = l_begin; l_id < l_end; l_id++ ){
      int64_t l_id_sfc_m_new, l_id_sfc_n_new, l_id_omp_new; 
      sfc_oracle_2d( &l_id_sfc_m_new, &l_id_sfc_n_new, &l_id_omp_new, l_id+1 );

      if( l_id_omp_new != l_id_omp_old ){
        sfc_t l_move = get_max_dim_jump( m_omp_loops, l_id_omp_new, l_id_omp_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move; 
      }
      else if( l_id_sfc_m_new != l_id_sfc_m_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_m, l_id_sfc_m_new, l_id_sfc_m_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }
      else if( l_id_sfc_n_new != l_id_sfc_n_old ){
        sfc_t l_move = get_max_dim_jump( m_sfc_loops_n, l_id_sfc_n_new, l_id_sfc_n_old );
        m_dim_movements[l_thread_id][l_id-l_begin] = l_move;
      }

      l_id_sfc_m_old = l_id_sfc_m_new;
      l_id_sfc_n_old = l_id_sfc_n_new;
      l_id_omp_old = l_id_omp_new;
    }
  }

  return err_t::SUCCESS;
}

int64_t einsum_ir::binary::IterationSpace::calculate_offset( int64_t i_id_omp,
                                                             int64_t i_id_sfc_m,
                                                             int64_t i_id_sfc_n,
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
  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    l_offset += (i_id_omp % l_size) * l_stride;
    i_id_omp /= l_size;
  }

  return l_offset;
}



void einsum_ir::binary::IterationSpace::convert_strides_to_offsets( std::vector< int64_t > const & i_strides,
                                                                    std::vector< int64_t >       & io_offsets ) {
   
  int64_t l_first = m_parallel_loops.begin;

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

  int64_t l_id_sfc_m, l_id_sfc_n, l_id_omp;
  sfc_oracle_2d(&l_id_sfc_m, &l_id_sfc_n, &l_id_omp, m_sfc_tasks_m*m_sfc_tasks_n-1);
  int64_t l_all_offsets_omp = calculate_offset( l_id_omp,
                                                l_id_sfc_m,
                                                l_id_sfc_n,
                                                i_strides );

  for (int64_t l_id = m_omp_loops.end - 1; l_id >= m_omp_loops.begin; l_id--) {
    int64_t l_size   = m_loop_sizes->at(l_id);
    int64_t l_stride = i_strides[l_id];

    io_offsets[ l_id - l_first] = l_stride - l_all_offsets_omp;
    l_all_offsets_omp += (l_size - 1) * l_stride;
  }
}

einsum_ir::binary::sfc_t einsum_ir::binary::IterationSpace::get_max_dim_jump( range_t i_dim_loops,
                                                                              int64_t i_id_new,
                                                                              int64_t i_id_old ){

  int64_t l_direction = (( i_id_old - i_id_new ) + 1) / 2;
  int64_t l_max_id = i_id_new > i_id_old ? i_id_new : i_id_old;
  for( int64_t l_di = i_dim_loops.end-1; l_di >= i_dim_loops.begin; l_di-- ){
    int64_t l_size = m_loop_sizes->at(l_di);
    if(l_max_id % l_size != 0){
      return (l_di - m_parallel_loops.begin) * 2 + l_direction;
    }
    else{
      l_max_id /= l_size;
    }
  }

  return 0;
}

int64_t einsum_ir::binary::IterationSpace::get_num_tasks( int64_t i_thread_id ){
  return m_dim_movements[i_thread_id].size();
}

void einsum_ir::binary::IterationSpace::add_movement_offsets( int64_t          i_thread_id, 
                                                              int64_t          i_task_id,
                                                              char    const ** io_ptr_left,
                                                              char    const ** io_ptr_right,
                                                              char    const ** io_ptr_out_aux,
                                                              char          ** io_ptr_out){
  sfc_t l_move =  m_dim_movements[i_thread_id][i_task_id];
  sfc_t l_sign = (l_move & 1);
  int64_t l_direction = 1 - ( (int64_t)l_sign << 1); 
  l_move = l_move >> 1;

  *io_ptr_left    += l_direction * m_movement_offsets[0][l_move];
  *io_ptr_right   += l_direction * m_movement_offsets[1][l_move];
  *io_ptr_out     += l_direction * m_movement_offsets[2][l_move];
  *io_ptr_out_aux += l_direction * m_movement_offsets[3][l_move];
}

void einsum_ir::binary::IterationSpace::get_initial_offsets( int64_t   i_thread_id,
                                                             int64_t & o_off_left,
                                                             int64_t & o_off_right,
                                                             int64_t & o_off_out_aux,
                                                             int64_t & o_off_out ) {
  o_off_left    = m_initial_offsets[i_thread_id][0];
  o_off_right   = m_initial_offsets[i_thread_id][1];
  o_off_out     = m_initial_offsets[i_thread_id][2];
  o_off_out_aux = m_initial_offsets[i_thread_id][3];
}

void einsum_ir::binary::IterationSpace::sfc_oracle_2d( int64_t *i_m, 
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
