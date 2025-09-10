import os, sys

print( 'running build script of einsum_ir' )

# configuration
l_vars = Variables()

l_vars.AddVariables(
  EnumVariable( 'mode',
                'compile modes, option \'san\' enables address and undefined behavior sanitizers',
                'release',
                 allowed_values=('release', 'debug', 'release+san', 'debug+san' )
              ),
  EnumVariable( 'parallel',
                'used parallelization',
                'omp',
                 allowed_values=('none', 'omp') ),
  PackageVariable( 'libxsmm',
                   'Enable libxsmm backend.',
                   'yes' ),
  PackageVariable( 'blas',
                   'Enable BLAS backend.',
                   'yes' ),
  PackageVariable( 'tblis',
                   'Enable TBLIS backend.',
                   'yes' ),
  PackageVariable( 'libtorch',
                   'Enable libtorch.',
                   'no' )
)

# create environment
g_env = Environment( variables = l_vars )

# include environment
g_env['OS_ENV'] = os.environ

# exit in the case of unknown variables
if l_vars.UnknownVariables():
  print( "build configuration corrupted, don't know what to do with: " + str(l_vars.UnknownVariables().keys()) )
  exit(1)

# generate help message
Help( l_vars.GenerateHelpText(g_env) )

# configuration
g_conf = Configure( g_env )

# forward compiler
if 'CC' in g_env['OS_ENV'].keys():
  g_env['CC'] = g_env['OS_ENV']['CC']
if 'CXX' in g_env['OS_ENV'].keys():
  g_env['CXX'] = g_env['OS_ENV']['CXX']

# set optimization mode
if 'debug' in g_env['mode']:
  g_env.AppendUnique( CXXFLAGS = [ '-g',
                                   '-O0' ] )
  # set strict warnings
  g_env.AppendUnique( CXXFLAGS = [ '-Wall',
                                   '-Wextra',
                                   '-Wcast-align',
                                   '-pedantic',
                                   '-Wshadow',
                                   '-Wdisabled-optimization',
                                   '-Wduplicated-branches',
                                   '-Wduplicated-cond',
                                   '-Wlogical-op',
                                   '-Wnull-dereference',
                                   '-Woverloaded-virtual',
                                   '-Wpointer-arith',
                                   '-Wshadow' ] )
  # exceptions for annoying warnings
  g_env.AppendUnique( CXXFLAGS = [ '-Wno-comment' ] )
else:
  g_env.Append( CPPDEFINES = ['PP_NDEBUG'] )
  g_env.Append( CXXFLAGS = ['-O2'] )
# add sanitizers
if 'san' in  g_env['mode']:
  g_env.AppendUnique( CXXFLAGS =  [ '-g',
                                    '-fsanitize=float-divide-by-zero',
                                    '-fsanitize=address',
                                    '-fsanitize=undefined',
                                    '-fno-omit-frame-pointer',
                                    '-fsanitize=pointer-compare',
                                    '-fsanitize=pointer-subtract'] )

  if sys.platform != 'darwin':
    g_env.AppendUnique( CXXFLAGS = [ '-fsanitize=leak' ] )

  g_env.AppendUnique( LINKFLAGS = [ '-g',
                                    '-fsanitize=address',
                                    '-fsanitize=undefined'] )

# enable c++17
if g_env['CXX'].startswith("g++"):
  g_env.AppendUnique( CXXFLAGS = [ '-std=gnu++17' ] )
else:
  g_env.AppendUnique( CXXFLAGS = [ '-std=c++17' ] )

# enable omp
if 'omp' in g_env['parallel']:
  g_env.AppendUnique( CPPFLAGS = ['-fopenmp'] )
  g_env.AppendUnique( CPPFLAGS = ['-fopenmp-simd'] )
  g_env.AppendUnique( LINKFLAGS = ['-fopenmp'] )

# discover libraries
if g_env['libtorch'] != False:
  if g_env['libtorch'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libtorch'] + '/include') ] )
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libtorch'] + '/include/torch/csrc/api/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['libtorch'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['libtorch'] + '/lib'] )
    try:
      with open( g_env['libtorch'] + '/share/cmake/Torch/TorchConfig.cmake' ) as l_file:
        l_contents = l_file.read()
        if( '-D_GLIBCXX_USE_CXX11_ABI=0' in l_contents ):
          g_env.AppendUnique( CPPDEFINES = ['_GLIBCXX_USE_CXX11_ABI=0'] )
    except:
      pass

  if not( g_conf.CheckLib( 'libc10',
                           language='CXX' ) and \
          g_conf.CheckLib( 'libtorch_cpu',
                            language='CXX' ) and \
          g_conf.CheckLibWithHeader( 'libtorch',
                                     ['ATen/ATen.h', 'torch/torch.h'],
                                     'CXX' ) ):
    g_env['libtorch'] = False

if g_env['libxsmm'] != False:
  if g_env['libxsmm'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libxsmm'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['libxsmm'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['libxsmm'] + '/lib'] )

    if not g_conf.CheckLibWithHeader( 'libxsmm',
                                      'libxsmm.h',
                                      'CXX' ):
      g_env['libxsmm'] = False

g_env['blas_has_imatcopy'] = False

if g_env['blas'] != False:
  if g_env['blas'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['blas'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['blas'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['blas'] + '/lib'] )

  # try to discover NVPL BLAS
  if g_conf.CheckLibWithHeader( 'nvpl_blas_lp64_seq',
                                'nvpl_blas_cblas.h',
                                'CXX' ):
     g_env['blas'] = 'nvpl'
  # try to discover accelerate
  elif( g_env['blas'] == True and g_env['HOST_OS'] == "darwin" ):
    g_env.AppendUnique( CXXFLAGS  = [ '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/' ] )
    g_env.AppendUnique( LINKFLAGS = [ '-framework',  'Accelerate' ] )
    g_env['blas'] = 'accelerate'
  # try to discover openblas
  elif g_conf.CheckLibWithHeader( 'openblas',
                                  'cblas.h',
                                  'CXX' ):
    g_env['blas'] = 'openblas'

  # check if the required BLAS routines (sgemm, dgemm) and extensiosn (simatcopy, dimatcopy) are available
  if    not g_conf.CheckFunc('cblas_sgemm',     language='CXX') \
     or not g_conf.CheckFunc('cblas_dgemm',     language='CXX'):
    g_env['blas'] = False

  if     g_conf.CheckFunc('cblas_simatcopy', language='CXX') \
     and g_conf.CheckFunc('cblas_dimatcopy', language='CXX'):
    g_env['blas_has_imatcopy'] = True

if g_env['tblis'] != False:
  if g_env['tblis'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['tblis'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['tblis'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['tblis'] + '/lib'] )

  # try to discover tci
  g_env['tci'] = g_conf.CheckLibWithHeader( 'tci',
                                            'tci.h',
                                            'CXX' )

  # try to discover tblis
  g_env['tblis'] = g_conf.CheckLibWithHeader( 'tblis',
                                              'tblis/tblis.h',
                                              'CXX' )

# macOS: convert every RPATH entry into an explicit link flag
if sys.platform == "darwin" and g_env.get('RPATH'):
  rpath_flags = [f"-Wl,-rpath,{p}" for p in g_env['RPATH']]
  g_env.AppendUnique(LINKFLAGS=rpath_flags)

# build
g_env['build_dir'] = 'build'

g_env.AppendUnique( CPPPATH = [ '#.' ] )
g_env.AppendUnique( CPPPATH = [ '#/src' ] )

# get source files
VariantDir( g_env['build_dir']+'/src', 'src')

g_env.sources = []
g_env.tests = []
g_env.exe = {}

Export('g_env')
SConscript( g_env['build_dir']+'/src/basic/SConscript' )
SConscript( g_env['build_dir']+'/src/SConscript' )
Import('g_env')

if( g_env['libxsmm'] and g_env['libtorch'] ):
  g_env.Program( g_env['build_dir']+'/bench_unary',
                 source = g_env.sources + g_env.exe['bench_unary'] )
  g_env.Program( g_env['build_dir']+'/bench_binary',
                 source = g_env.sources + g_env.exe['bench_binary'] )
  g_env.Program( g_env['build_dir']+'/bench_expression',
                 source = g_env.sources + g_env.exe['bench_expression'] )
  g_env.Program( g_env['build_dir']+'/bench_mlp',
                 source = g_env.sources + g_env.exe['bench_mlp'] )
  g_env.Program( g_env['build_dir']+'/bench_tree',
                 source = g_env.sources + g_env.exe['bench_tree'] )

g_env.Program( g_env['build_dir']+'/tests',
               source = g_env.tests + g_env.sources )