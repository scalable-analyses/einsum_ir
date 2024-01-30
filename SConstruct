import os

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
                   'Enable libxsmm.',
                   'no' ),
  PackageVariable( 'blas',
                   'Enable BLAS.',
                   'no' ),
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
                                    '-fsanitize=pointer-subtract',
                                    '-fsanitize=leak'] )
  g_env.AppendUnique( LINKFLAGS = [ '-g',
                                    '-fsanitize=address',
                                    '-fsanitize=undefined'] )

# enable c++17
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
          g_env.AppendUnique( CPPDEFINES='_GLIBCXX_USE_CXX11_ABI=0' )
    except:
      pass

  if not( g_conf.CheckLib( 'libc10',
                           language='CXX' ) and \
          g_conf.CheckLib( 'libtorch_cpu',
                            language='CXX' ) and \
          g_conf.CheckLibWithHeader( 'libtorch',
                                     ['ATen/ATen.h', 'torch/torch.h'],
                                     'CXX' ) ):
    print( 'warning: disabling libtorch' )
    g_env['libtorch'] = False

if g_env['libxsmm'] != False:
  if g_env['libxsmm'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['libxsmm'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['libxsmm'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['libxsmm'] + '/lib'] )

    if not g_conf.CheckLibWithHeader( 'libxsmm',
                                      'libxsmm.h',
                                      'CXX' ):
     print( 'warning: disabling libxsmm' )
     g_env['libxsmm'] = False

if g_env['blas'] != False:
  if g_env['blas'] != True:
    g_env.AppendUnique( CXXFLAGS = [ ('-isystem',  g_env['blas'] + '/include') ] )
    g_env.AppendUnique( LIBPATH = [ g_env['blas'] + '/lib'] )
    g_env.AppendUnique( RPATH = [ g_env['blas'] + '/lib'] )

  # try to discover accelerate
  if( g_env['blas'] == True and g_env['HOST_OS'] == "darwin" ):
    g_env.AppendUnique( CXXFLAGS  = [ '-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/' ] )
    g_env.AppendUnique( LINKFLAGS = [ '-framework',  'Accelerate' ] )
  # try to discover openblas
  else:
    g_conf.CheckLibWithHeader( 'openblas',
                               'cblas.h',
                               'CXX' )

  # check if the required BLAS routines (sgemm, dgemm) and extensiosn (simatcopy, dimatcopy) are available
  if    not g_conf.CheckFunc('cblas_sgemm',     language='CXX') \
     or not g_conf.CheckFunc('cblas_dgemm',     language='CXX') \
     or not g_conf.CheckFunc('cblas_simatcopy', language='CXX') \
     or not g_conf.CheckFunc('cblas_dimatcopy', language='CXX'):
    print( 'warning: disabling blas' )
    g_env['blas'] = False

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
  g_env.Program( g_env['build_dir']+'/bench_resnet',
                 source = g_env.sources + g_env.exe['bench_resnet'] )

g_env.Program( g_env['build_dir']+'/tests',
               source = g_env.tests )