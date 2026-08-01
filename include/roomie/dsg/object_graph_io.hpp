#pragma once

#include <filesystem>
#include <string>

#include "roomie/dsg/object_graph.hpp"

namespace roomie {

struct ObjectGraphSavePaths {
  std::filesystem::path primary_path;
  std::filesystem::path latest_path;
  std::filesystem::path snapshot_image_dir;
  std::string snapshot_uri_prefix;
};

bool resolveObjectGraphSavePaths(const std::string& configured_path,
                                 TimeNanoseconds saved_time_ns,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error);

bool resolveObjectGraphSavePaths(const std::string& configured_path,
                                 TimeNanoseconds saved_time_ns,
                                 const std::string& snapshot_image_subdir,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error);

bool saveObjectGraphSnapshotJsonAtomic(const ObjectGraphSnapshot& snapshot,
                                       const std::string& world_frame,
                                       TimeNanoseconds saved_time_ns,
                                       const std::filesystem::path& path,
                                       std::string* error);

bool saveObjectGraphSnapshotJson(const ObjectGraphSnapshot& snapshot,
                                 const std::string& world_frame,
                                 TimeNanoseconds saved_time_ns,
                                 const std::string& configured_path,
                                 ObjectGraphSavePaths* paths,
                                 std::string* error);

bool loadObjectGraphSnapshotJson(const std::filesystem::path& path,
                                 ObjectGraphSnapshot* snapshot,
                                 std::string* world_frame,
                                 std::string* error);

}  // namespace roomie
