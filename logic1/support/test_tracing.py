import io

import pytest

from logic1.support.tracing import trace


def test_trace_call_and_return():
    stream = io.StringIO()

    @trace(stream=stream)
    def foo():
        return 1

    assert foo() == 1
    assert stream.getvalue() == (
        "--> test_trace_call_and_return.<locals>.foo()\n"
        "<-- test_trace_call_and_return.<locals>.foo == 1\n"
    )


def test_trace_arguments():
    stream = io.StringIO()

    @trace(stream=stream)
    def foo(a, b=2):
        return a + b

    assert foo(1, b=3) == 4
    assert stream.getvalue() == (
        "--> test_trace_arguments.<locals>.foo(1, b=3)\n"
        "<-- test_trace_arguments.<locals>.foo == 4\n"
    )


def test_trace_nested_calls():
    stream = io.StringIO()

    @trace(stream=stream)
    def inner():
        return 1

    @trace(stream=stream)
    def outer():
        return inner()

    assert outer() == 1
    assert stream.getvalue() == (
        "--> test_trace_nested_calls.<locals>.outer()\n"
        "  --> test_trace_nested_calls.<locals>.inner()\n"
        "  <-- test_trace_nested_calls.<locals>.inner == 1\n"
        "<-- test_trace_nested_calls.<locals>.outer == 1\n"
    )


def test_trace_indent_step():
    stream = io.StringIO()

    @trace(stream=stream, indent_step=4)
    def outer():
        return inner()

    @trace(stream=stream, indent_step=4)
    def inner():
        return 1

    assert outer() == 1
    assert stream.getvalue() == (
        "--> test_trace_indent_step.<locals>.outer()\n"
        "    --> test_trace_indent_step.<locals>.inner()\n"
        "    <-- test_trace_indent_step.<locals>.inner == 1\n"
        "<-- test_trace_indent_step.<locals>.outer == 1\n"
    )


def test_trace_show_ret_false():
    stream = io.StringIO()

    @trace(stream=stream, show_ret=False)
    def foo():
        return 1

    assert foo() == 1
    assert stream.getvalue() == (
        "--> test_trace_show_ret_false.<locals>.foo()\n"
    )


def test_trace_str():
    stream = io.StringIO()

    @trace(stream=stream, str=True)
    def foo(value):
        return value

    assert foo("abc") == "abc"
    assert stream.getvalue() == (
        "--> test_trace_str.<locals>.foo(abc)\n"
        "<-- test_trace_str.<locals>.foo == abc\n"
    )


def test_trace_pretty():
    stream = io.StringIO()

    @trace(stream=stream, pretty=True)
    def foo(value):
        return value

    value = {"b": 2, "a": [1, 2]}
    assert foo(value) == value
    assert stream.getvalue() == (
        "--> test_trace_pretty.<locals>.foo({'a': [1, 2], 'b': 2})\n"
        "<-- test_trace_pretty.<locals>.foo == {'a': [1, 2], 'b': 2}\n"
    )


def test_trace_indent_restored_after_exception():
    stream = io.StringIO()

    @trace(stream=stream)
    def bad():
        raise ValueError("boom")

    @trace(stream=stream)
    def good():
        return 1

    with pytest.raises(ValueError, match="boom"):
        bad()

    assert good() == 1
    assert stream.getvalue() == (
        "--> test_trace_indent_restored_after_exception.<locals>.bad()\n"
        "--> test_trace_indent_restored_after_exception.<locals>.good()\n"
        "<-- test_trace_indent_restored_after_exception.<locals>.good == 1\n"
    )


def test_trace_indent_restored_after_nested_exception():
    stream = io.StringIO()

    @trace(stream=stream)
    def inner():
        raise ValueError("boom")

    @trace(stream=stream)
    def outer():
        inner()

    with pytest.raises(ValueError, match="boom"):
        outer()

    assert trace.cur_indent == 0
    assert stream.getvalue() == (
        "--> test_trace_indent_restored_after_nested_exception.<locals>.outer()\n"
        "  --> test_trace_indent_restored_after_nested_exception.<locals>.inner()\n"
    )


def test_trace_does_not_add_an_extra_blank_line():
    stream = io.StringIO()

    @trace(stream=stream)
    def foo():
        return "result"

    foo()

    assert stream.getvalue().splitlines() == [
        "--> test_trace_does_not_add_an_extra_blank_line.<locals>.foo()",
        "<-- test_trace_does_not_add_an_extra_blank_line.<locals>.foo == 'result'",
    ]
    assert "\n\n" not in stream.getvalue()
