import pytest

from onnxruntime.capi.training import model_desc_validation as md_val


@pytest.mark.parametrize("test_input", [
    ({'inputs': [('in0', [])],
      'outputs': [('out0', []), ('out1', [])]}),
    ({'inputs': [('in0', ['batch', 2, 3])],
      'outputs': [('out0', [], True)]}),
    ({'inputs': [('in0', []), ('in1', [1]), ('in2', [1, 2]), ('in3', [1000, 'dyn_ax1']), ('in4', ['dyn_ax1', 'dyn_ax2', 'dyn_ax3'])],
      'outputs': [('out0', [], True), ('out1', [1], False), ('out2', [1, 'dyn_ax1', 3])]})
])
def testORTTrainerModelDescValidSchemas(test_input):
    r''' Test different ways of using default values for incomplete input'''
    md_val._ORTTrainerModelDesc(test_input)


@pytest.mark.parametrize("test_input,error_msg", [
    ({'inputs': [(True, [])],
      'outputs': [(True, [])]},
     "Invalid model_desc: {'inputs': [{0: ['the first element of the tuple (aka name) must be a string']}], 'outputs': [{0: ['the first element of the tuple (aka name) must be a string']}]}"),
    ({'inputs': [('in1', None)],
      'outputs': [('out1', None)]},
     "Invalid model_desc: {'inputs': [{0: ['the second element of the tuple (aka shape) must be a list']}], 'outputs': [{0: ['the second element of the tuple (aka shape) must be a list']}]}"),
    ({'inputs': [('in1', [])],
      'outputs': [('out1', [], None)]},
     "Invalid model_desc: {'outputs': [{0: ['the third element of the tuple (aka is_loss) must be a boolean']}]}"),
    ({'inputs': [('in1', [True])],
      'outputs': [('out1', [True])]},
     "Invalid model_desc: {'inputs': [{0: ['each shape must be either a string or integer']}], 'outputs': [{0: ['each shape must be either a string or integer']}]}"),
    ({'inputs': [('in1', [])],
      'outputs': [('out1', [], True), ('out2', [], True)]},
     "Invalid model_desc: {'outputs': [{1: ['only one is_loss can bet set to True']}]}"),
])
def testORTTrainerModelDescInvalidSchemas(test_input, error_msg):
    r''' Test different ways of using default values for incomplete input'''
    with pytest.raises(ValueError) as e:
        md_val._ORTTrainerModelDesc(test_input)
    assert str(e.value) == error_msg
