---@diagnostic disable: undefined-global
local lib_type = 'static'
local opencv_home = string.format('%s/%s', os.getenv('OPENCV_HOME'), lib_type)
local opencv_include_dir = opencv_home .. '/include'
local opencv_link_dir = opencv_home .. '/x64/mingw/lib'
add_rules("mode.debug", "mode.release")

target("cpp")
    set_kind("binary")
    add_files("src/main.cpp")
    add_files("src/radon_transform.cpp")
    add_files("src/graph.cpp")
    add_includedirs('./include')
    -- opencv
    add_includedirs(opencv_include_dir)
    add_linkdirs(opencv_link_dir)
    add_links('opencv_world470')
    if lib_type == 'static' then
        add_linkdirs([[D:\Softwares\Program_Files\C\mingw64\x86_64-w64-mingw32\lib]])
        add_links(
            'gdi32',
            'ComDlg32',
            'OleAut32',
            'Ole32',
            'uuid',
            'libpng',
            'libopenjp2',
            'ade',
            'IlmImf',
            'libjpeg-turbo',
            'libprotobuf',
            'libtiff',
            'libwebp',
            'quirc',
            'zlib',
            'opencv_img_hash470'
        )
    end

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

