# gen-hmm

Generates HMMs for the Advanced AI course.

## Domain

This is an object tracking problem where some target travels forward
in a grid. Observations are also positions, but are noisy and can be
adjacent cells.

## Usage

The domain doesn't really matter. All state and observation
probabilities are specified in the GridHMM object, through the
`initial_p`, `transition_p` and `observation_p` methods. You should
design your inference mechanism to be generalizable to any transition
matrix, so the domain actually doesn't really matter at all.

Also, read the code.
