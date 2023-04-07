#include "pybind11/pybind11.h"
#include "pybind11/gil.h"
#include "pygil.hpp"


PyGilSwitcher::PyGilSwitcher()
{
    pybind11::gil_scoped_release release_gil;
}


PyGilSwitcher::~PyGilSwitcher()
{
    pybind11::gil_scoped_acquire acquire_gil;
}
