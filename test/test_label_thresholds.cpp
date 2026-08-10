#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "roomie/pipeline/label_thresholds.hpp"

namespace roomie {
namespace {

class TemporaryJsonFile {
 public:
  explicit TemporaryJsonFile(const std::string& contents) {
    char path[] = "/tmp/roomie_label_thresholds_XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
      throw std::runtime_error("mkstemp failed");
    }
    ::close(fd);
    path_ = path;
    std::ofstream stream(path_);
    stream << contents;
    if (!stream) {
      throw std::runtime_error("failed to write temporary JSON file");
    }
  }

  ~TemporaryJsonFile() { ::unlink(path_.c_str()); }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

TEST(LabelConfidenceThresholds, LoadsOverridesAndUsesFallback) {
  TemporaryJsonFile file(
      R"({"instance":{"default":0.35,"labels":{"chair":0.2,"table":0.75}}})");
  const LabelConfidenceThresholds thresholds =
      loadLabelConfidenceThresholds(file.path(), "instance");

  EXPECT_FLOAT_EQ(labelConfidenceThreshold(thresholds, "chair", 0.5f), 0.2f);
  EXPECT_FLOAT_EQ(labelConfidenceThreshold(thresholds, "table", 0.5f), 0.75f);
  EXPECT_FLOAT_EQ(labelConfidenceThreshold(thresholds, "lamp", 0.5f), 0.35f);
}

TEST(LabelConfidenceThresholds, EmptyPathKeepsGlobalDefaults) {
  const LabelConfidenceThresholds thresholds =
      loadLabelConfidenceThresholds("", "instance");
  EXPECT_TRUE(thresholds.by_label.empty());
  EXPECT_FALSE(thresholds.configured_default.has_value());
  EXPECT_FLOAT_EQ(labelConfidenceThreshold(thresholds, "chair", 0.35f),
                  0.35f);
}

TEST(LabelConfidenceThresholds, RejectsOutOfRangeValues) {
  TemporaryJsonFile file(
      R"({"instance":{"default":0.35,"labels":{"chair":1.1}}})");
  EXPECT_THROW(loadLabelConfidenceThresholds(file.path(), "instance"),
               std::runtime_error);
}

TEST(LabelConfidenceThresholds, RejectsNonObjectDocuments) {
  TemporaryJsonFile file(R"([0.25, 0.35])");
  EXPECT_THROW(loadLabelConfidenceThresholds(file.path(), "instance"),
               std::runtime_error);
}

TEST(LabelConfidenceThresholds, RequiresDefaultAndLabelsFields) {
  TemporaryJsonFile missing_default(R"({"instance":{"labels":{}}})");
  EXPECT_THROW(
      loadLabelConfidenceThresholds(missing_default.path(), "instance"),
      std::runtime_error);

  TemporaryJsonFile missing_labels(R"({"instance":{"default":0.35}})");
  EXPECT_THROW(
      loadLabelConfidenceThresholds(missing_labels.path(), "instance"),
      std::runtime_error);
}

}  // namespace
}  // namespace roomie
