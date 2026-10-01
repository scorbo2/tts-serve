"""Import-only stand-in for ``breeze_infer.templates`` (test machines only).

The template-selection / input-preparation functions raise if a test
reaches them without monkeypatching the server's module-level names — the
success-path tests install their own recording stand-ins.
"""


def select_template_name(request):
    raise NotImplementedError(
        "breeze_infer stub: select_template_name() is not available in tests"
    )


def get_template(name):
    raise NotImplementedError(
        "breeze_infer stub: get_template() is not available in tests"
    )


def prepare_inputs(*args, **kwargs):
    raise NotImplementedError(
        "breeze_infer stub: prepare_inputs() is not available in tests"
    )
