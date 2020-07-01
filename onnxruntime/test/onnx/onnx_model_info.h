// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "TestCase.h"

class OnnxModelInfo : public TestModelInfo {
 private:
  std::string node_name_;
  std::string onnx_commit_tag_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> input_value_info_;
  std::vector<ONNX_NAMESPACE::ValueInfoProto> output_value_info_;

  const std::basic_string<PATH_CHAR_TYPE> model_url_;

 public:
  OnnxModelInfo(_In_ const PATH_CHAR_TYPE* model_url);
  
  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_url_.c_str(); }
  std::string GetModelVersion() const override { return onnx_commit_tag_; }

  const std::string& GetNodeName() const override { return node_name_; }
  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return &output_value_info_[i];
  }
  int GetInputCount() const override { return static_cast<int>(input_value_info_.size()); }
  int GetOutputCount() const override { return static_cast<int>(output_value_info_.size()); }
  const std::string& GetInputName(size_t i) const override { return input_value_info_[i].name(); }

  const std::string& GetOutputName(size_t i) const override { return output_value_info_[i].name(); }
};