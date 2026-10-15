# Writes the iPIC3D build identity to a C header (see src/include/BuildInfo.h).
#
# Run as a script at configure time and before every build:
#   cmake -DSOURCE_DIR=<repo> -DBINARY_DIR=<build dir> -DOUTPUT=<header>
#         -DBUILD_TYPE=<type> -DBUILD_OPTIONS=<"NAME=VALUE ...">
#         -DCOMPILERS=<text>
#         -P GenerateBuildInfo.cmake
# The header is rewritten only when its content changes, so an unchanged tree
# does not trigger recompilation. A tree with uncommitted changes (modified,
# staged or untracked files that git does not ignore) is reported with a
# warning, and a hash of those changes tells such builds apart. Untracked files
# inside the build directory are build products, not changes.

set(git_commit "unknown")
set(git_describe "unknown")
set(git_branch "unknown")
set(git_date "unknown")
set(git_dirty -1)
set(git_dirty_files 0)
set(git_dirty_hash "")

find_program(GIT_EXECUTABLE git)
if(GIT_EXECUTABLE)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE git_rc
    OUTPUT_VARIABLE git_commit_out
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

if(GIT_EXECUTABLE AND git_rc EQUAL 0)
  set(git_commit "${git_commit_out}")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" describe --tags --always
            --abbrev=10
    OUTPUT_VARIABLE git_describe
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" rev-parse --abbrev-ref HEAD
    OUTPUT_VARIABLE git_branch
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" log -1 --format=%cI
    OUTPUT_VARIABLE git_date
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

  # Leave out the build directory when it lies inside the source tree and
  # .gitignore does not cover it: its files, the generated header among them,
  # would change the hash on every build. An in-source build cannot tell build
  # products from new source files, so there only tracked files count.
  set(untracked_mode all)
  set(pathspec .)
  if(BINARY_DIR AND EXISTS "${BINARY_DIR}")
    file(REAL_PATH "${SOURCE_DIR}" source_real)
    file(REAL_PATH "${BINARY_DIR}" binary_real)
    file(RELATIVE_PATH binary_rel "${source_real}" "${binary_real}")
    if(binary_rel STREQUAL "")
      set(untracked_mode no)
    elseif(NOT binary_rel MATCHES "^\\.\\.(/|$)"
           AND NOT IS_ABSOLUTE "${binary_rel}")
      list(APPEND pathspec ":(exclude,literal)${binary_rel}")
    endif()
  endif()

  # Uncommitted state: everything git status reports (it honours .gitignore).
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" status --porcelain
            --untracked-files=${untracked_mode} -- ${pathspec}
    OUTPUT_VARIABLE git_status
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(git_status STREQUAL "")
    set(git_dirty 0)
  else()
    set(git_dirty 1)
    string(REPLACE "\n" ";" status_lines "${git_status}")
    list(LENGTH status_lines git_dirty_files)

    # Hash of the changes: the diff against HEAD and the content of untracked
    # files, so two different modifications of one commit differ. Untracked
    # files above 1 MB (data, not code) enter with their name and size only, so
    # they do not slow down every build.
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" diff HEAD --binary --
              ${pathspec}
      OUTPUT_VARIABLE git_diff ERROR_QUIET)
    set(untracked "")
    if(untracked_mode STREQUAL "all")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" ls-files --others
                --exclude-standard -- ${pathspec}
        OUTPUT_VARIABLE untracked
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
    set(state "${git_diff}")
    if(NOT untracked STREQUAL "")
      string(REPLACE "\n" ";" untracked "${untracked}")
      foreach(f IN LISTS untracked)
        if(EXISTS "${SOURCE_DIR}/${f}" AND NOT IS_DIRECTORY "${SOURCE_DIR}/${f}")
          file(SIZE "${SOURCE_DIR}/${f}" fsize)
          if(fsize LESS 1048576)
            file(SHA1 "${SOURCE_DIR}/${f}" fh)
          else()
            set(fh "size ${fsize}")
          endif()
          string(APPEND state "untracked ${f} ${fh}\n")
        endif()
      endforeach()
    endif()
    string(SHA1 state_hash "${state}")
    string(SUBSTRING "${state_hash}" 0 10 git_dirty_hash)

    # List at most 20 changed files in the warning.
    # (indented, so message() keeps them as one block)
    list(SUBLIST status_lines 0 20 shown)
    list(TRANSFORM shown PREPEND "  ")
    string(REPLACE ";" "\n" shown "${shown}")
    if(git_dirty_files GREATER 20)
      string(APPEND shown "\n ... and more")
    endif()
    message(WARNING
      "iPIC3D is being built from a working tree with uncommitted changes "
      "(${git_dirty_files} file(s), state ${git_dirty_hash}); the binary "
      "cannot be rebuilt from the commit alone.\n${shown}")
  endif()
endif()

# Text that ends up in C string literals and in JSON: no quotes or backslashes.
foreach(var git_commit git_describe git_branch git_date BUILD_TYPE
            BUILD_OPTIONS COMPILERS)
  string(REPLACE "\\" "/" ${var} "${${var}}")
  string(REPLACE "\"" "'" ${var} "${${var}}")
endforeach()
if(BUILD_TYPE STREQUAL "")
  set(BUILD_TYPE "none")
endif()

set(content "// Generated by cmake/GenerateBuildInfo.cmake; do not edit.
#pragma once
#define IPIC3D_GIT_COMMIT \"${git_commit}\"
#define IPIC3D_GIT_DESCRIBE \"${git_describe}\"
#define IPIC3D_GIT_BRANCH \"${git_branch}\"
#define IPIC3D_GIT_DATE \"${git_date}\"
#define IPIC3D_GIT_DIRTY ${git_dirty}
#define IPIC3D_GIT_DIRTY_FILES ${git_dirty_files}
#define IPIC3D_GIT_DIRTY_HASH \"${git_dirty_hash}\"
#define IPIC3D_BUILD_TYPE \"${BUILD_TYPE}\"
#define IPIC3D_BUILD_OPTIONS \"${BUILD_OPTIONS}\"
#define IPIC3D_COMPILERS \"${COMPILERS}\"
")

set(old "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" old)
endif()
if(NOT old STREQUAL content)
  file(WRITE "${OUTPUT}" "${content}")
endif()
