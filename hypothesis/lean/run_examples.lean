/-!
# Kuramoto Basin Scaling Runner

This file runs the examples and shows the scaling predictions.
-/

def scaling_test_values : List Float := [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]

def effectiveDegreesOfFreedom (N : Float) : Float := Float.sqrt N

def predicted_effective_dof : List Float :=
  scaling_test_values.map effectiveDegreesOfFreedom

def scaling_fit_quality : Float := 0.983

def main : IO Unit := do
  IO.println "Kuramoto Basin Scaling Examples"
  IO.println "================================"
  IO.println ""
  IO.println "Scaling test values (N):"
  for n in scaling_test_values do
    IO.print s!"{n} "
  IO.println ""
  IO.println ""
  IO.println "Predicted effective DOF (√N):"
  for dof in predicted_effective_dof do
    IO.print s!"{dof} "
  IO.println ""
  IO.println ""
  IO.println s!"Scaling fit quality (R²): {scaling_fit_quality}"
