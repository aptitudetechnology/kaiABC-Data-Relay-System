/-
Formalization of Three-Oscillator Kuramoto System with Isosceles Triangle Network

Based on: "Critical Points, Stability, and Basins of Attraction of Three Kuramoto
Oscillators with Isosceles Triangle Network"

This formalizes the specific N=3 case to test basin scaling hypotheses.
-/

-- Mathematical constants
def pi : Float := 3.141592653589793

-- Three-oscillator state structure
structure Kuramoto3State where
  θ1 : Float
  θ2 : Float
  θ3 : Float

-- Coupling parameters for isosceles triangle network
structure CouplingParams where
  K1 : Float  -- Coupling between oscillators 1-2 and 1-3
  K2 : Float  -- Coupling between oscillators 2-3

-- System dynamics for three oscillators with isosceles triangle topology
def kuramoto3Dynamics (state : Kuramoto3State) (params : CouplingParams) : Kuramoto3State :=
  let θ1 := state.θ1
  let θ2 := state.θ2
  let θ3 := state.θ3
  let K1 := params.K1
  let K2 := params.K2

  -- Equations from the paper (system 1):
  -- θ̇₁ = K₁ sin(θ₂ - θ₁) + K₁ sin(θ₃ - θ₁)
  -- θ̇₂ = K₁ sin(θ₁ - θ₂) + K₂ sin(θ₃ - θ₂)
  -- θ̇₃ = K₁ sin(θ₁ - θ₃) + K₂ sin(θ₂ - θ₃)
  { θ1 := K1 * Float.sin(θ2 - θ1) + K1 * Float.sin(θ3 - θ1),
    θ2 := K1 * Float.sin(θ1 - θ2) + K2 * Float.sin(θ3 - θ2),
    θ3 := K1 * Float.sin(θ1 - θ3) + K2 * Float.sin(θ2 - θ3) }

-- Order parameter for synchronization
def orderParameter3 (state : Kuramoto3State) : Float :=
  let r1 := Float.cos(state.θ1) + Float.cos(state.θ2) + Float.cos(state.θ3)
  let r2 := Float.sin(state.θ1) + Float.sin(state.θ2) + Float.sin(state.θ3)
  Float.sqrt(r1*r1 + r2*r2) / 3.0

-- Phase diameter function from the paper: 𝒟(Θ̃) = max(θᵢ) - min(θⱼ)
def phaseDiameter (state : Kuramoto3State) : Float :=
  let θs := [state.θ1, state.θ2, state.θ3]
  let maxθ := θs.foldl (fun acc x => if x > acc then x else acc) 0.0
  let minθ := θs.foldl (fun acc x => if x < acc then x else acc) pi
  maxθ - minθ

-- Critical points from Lemma 1
-- Basic critical points (always exist)
def criticalPoint1 : Kuramoto3State := {θ1 := 0.0, θ2 := 0.0, θ3 := 0.0}
def criticalPoint2 : Kuramoto3State := {θ1 := 0.0, θ2 := 0.0, θ3 := pi}
def criticalPoint3 : Kuramoto3State := {θ1 := pi, θ2 := pi, θ3 := pi}
def criticalPoint4 : Kuramoto3State := {θ1 := pi, θ2 := pi, θ3 := 0.0}

-- Conditional critical points (exist when K1 = -K2)
def criticalPoint5 (params : CouplingParams) : Option Kuramoto3State :=
  if params.K1 == -params.K2 then
    some { θ1 := 0.0, θ2 := 2.0 * pi / 3.0, θ3 := pi / 3.0 }
  else none

def criticalPoint6 (params : CouplingParams) : Option Kuramoto3State :=
  if params.K1 == -params.K2 then
    some { θ1 := pi, θ2 := pi / 3.0, θ3 := 2.0 * pi / 3.0 }
  else none

-- Jacobian matrix for stability analysis
def jacobian3 (state : Kuramoto3State) (params : CouplingParams) :
    (Float × Float × Float × Float × Float × Float × Float × Float × Float) :=
  let θ1 := state.θ1
  let θ2 := state.θ2
  let θ3 := state.θ3
  let K1 := params.K1
  let K2 := params.K2

  -- Partial derivatives for Jacobian matrix
  let J11 := -K1 * Float.cos(θ2 - θ1) - K1 * Float.cos(θ3 - θ1)
  let J12 := K1 * Float.cos(θ2 - θ1)
  let J13 := K1 * Float.cos(θ3 - θ1)

  let J21 := K1 * Float.cos(θ1 - θ2)
  let J22 := -K1 * Float.cos(θ1 - θ2) - K2 * Float.cos(θ3 - θ2)
  let J23 := K2 * Float.cos(θ3 - θ2)

  let J31 := K1 * Float.cos(θ1 - θ3)
  let J32 := K2 * Float.cos(θ2 - θ3)
  let J33 := -K1 * Float.cos(θ1 - θ3) - K2 * Float.cos(θ2 - θ3)

  (J11, J12, J13, J21, J22, J23, J31, J32, J33)

-- Basin of attraction regions from Theorem 1
def basinRegion5 (state : Kuramoto3State) : Bool :=
  let θ1_θ3 := state.θ1 - state.θ3
  let θ2_θ3 := state.θ2 - state.θ3
  let θ1_θ2 := state.θ1 - state.θ2

  -- Conditions: -π < θ₁(0) - θ₃(0) < π/3, -π/3 < θ₂(0) - θ₃(0) < π, -4π/3 < θ₁(0) - θ₂(0) < 0
  (-pi < θ1_θ3 ∧ θ1_θ3 < pi/3.0) ∧
  (-pi/3.0 < θ2_θ3 ∧ θ2_θ3 < pi) ∧
  (-4.0*pi/3.0 < θ1_θ2 ∧ θ1_θ2 < 0.0)

def basinRegion6 (state : Kuramoto3State) : Bool :=
  let θ1_θ3 := state.θ1 - state.θ3
  let θ2_θ3 := state.θ2 - state.θ3
  let θ1_θ2 := state.θ1 - state.θ2

  -- Conditions: -7π/3 < θ₁(0) - θ₃(0) < -π, -π < θ₂(0) - θ₃(0) < π/3, -2π < θ₁(0) - θ₂(0) < -2π/3
  (-7.0*pi/3.0 < θ1_θ3 ∧ θ1_θ3 < -pi) ∧
  (-pi < θ2_θ3 ∧ θ2_θ3 < pi/3.0) ∧
  (-2.0*pi < θ1_θ2 ∧ θ1_θ2 < -2.0*pi/3.0)

-- Test function to verify basin regions
def testBasinRegions : IO Unit := do
  let testStates := [
    {θ1 := 0.0, θ2 := 0.0, θ3 := 0.0},  -- Should be in region 5
    {θ1 := pi, θ2 := pi/3.0, θ3 := 2.0*pi/3.0},  -- Should be in region 6
    {θ1 := pi/2.0, θ2 := pi/2.0, θ3 := pi/2.0}   -- May not be in either
  ]

  for state in testStates do
    let in5 := basinRegion5 state
    let in6 := basinRegion6 state
    IO.println s!"State: θ1={state.θ1}, θ2={state.θ2}, θ3={state.θ3}"
    IO.println s!"  In basin 5: {in5}"
    IO.println s!"  In basin 6: {in6}"
    IO.println s!"  Order parameter: {orderParameter3 state}"
    IO.println s!"  Phase diameter: {phaseDiameter state}"
    IO.println ""

-- Main function for testing
def main : IO Unit := do
  IO.println "Three-Oscillator Kuramoto System Formalization"
  IO.println "=============================================="

  let params := {K1 := -1.0, K2 := 1.0}  -- K1 = -K2 = -1.0
  IO.println s!"Coupling parameters: K1={params.K1}, K2={params.K2}"

  -- Test critical points
  IO.println "\nCritical Points:"
  IO.println s!"Θ₁*: θ1={criticalPoint1.θ1}, θ2={criticalPoint1.θ2}, θ3={criticalPoint1.θ3}"
  IO.println s!"Θ₂*: θ1={criticalPoint2.θ1}, θ2={criticalPoint2.θ2}, θ3={criticalPoint2.θ3}"
  IO.println s!"Θ₃*: θ1={criticalPoint3.θ1}, θ2={criticalPoint3.θ2}, θ3={criticalPoint3.θ3}"
  IO.println s!"Θ₄*: θ1={criticalPoint4.θ1}, θ2={criticalPoint4.θ2}, θ3={criticalPoint4.θ3}"

  let cp5 := criticalPoint5 params
  let cp6 := criticalPoint6 params
  match cp5 with
  | some pt => IO.println s!"Θ₅*: θ1={pt.θ1}, θ2={pt.θ2}, θ3={pt.θ3}"
  | none => IO.println "Θ₅*: does not exist (K1 ≠ -K2)"

  match cp6 with
  | some pt => IO.println s!"Θ₆*: θ1={pt.θ1}, θ2={pt.θ2}, θ3={pt.θ3}"
  | none => IO.println "Θ₆*: does not exist (K1 ≠ -K2)"

  -- Test basin regions
  IO.println "\nTesting Basin Regions:"
  testBasinRegions

-- Mathematical constants
def pi : Float := 3.141592653589793

-- Three-oscillator state structure
structure Kuramoto3State where
  θ1 : Float
  θ2 : Float
  θ3 : Float

-- Coupling parameters for isosceles triangle network
structure CouplingParams where
  K1 : Float  -- Coupling between oscillators 1-2 and 1-3
  K2 : Float  -- Coupling between oscillators 2-3

-- System dynamics for three oscillators with isosceles triangle topology
def kuramoto3Dynamics (state : Kuramoto3State) (params : CouplingParams) : Kuramoto3State :=
  let θ1 := state.θ1
  let θ2 := state.θ2
  let θ3 := state.θ3
  let K1 := params.K1
  let K2 := params.K2

  -- Equations from the paper (system 1):
  -- θ̇₁ = K₁ sin(θ₂ - θ₁) + K₁ sin(θ₃ - θ₁)
  -- θ̇₂ = K₁ sin(θ₁ - θ₂) + K₂ sin(θ₃ - θ₂)
  -- θ̇₃ = K₁ sin(θ₁ - θ₃) + K₂ sin(θ₂ - θ₃)
  { θ1 := K1 * Float.sin(θ2 - θ1) + K1 * Float.sin(θ3 - θ1)
    θ2 := K1 * Float.sin(θ1 - θ2) + K2 * Float.sin(θ3 - θ2)
    θ3 := K1 * Float.sin(θ1 - θ3) + K2 * Float.sin(θ2 - θ3) }

-- Order parameter for synchronization
def orderParameter3 (state : Kuramoto3State) : Float :=
  let r1 := Float.cos(state.θ1) + Float.cos(state.θ2) + Float.cos(state.θ3)
  let r2 := Float.sin(state.θ1) + Float.sin(state.θ2) + Float.sin(state.θ3)
  Float.sqrt(r1*r1 + r2*r2) / 3.0

-- Phase diameter function from the paper: 𝒟(Θ̃) = max(θᵢ) - min(θⱼ)
def phaseDiameter (state : Kuramoto3State) : Float :=
  let θs := [state.θ1, state.θ2, state.θ3]
  let maxθ := θs.foldl (fun acc x => if x > acc then x else acc) 0.0
  let minθ := θs.foldl (fun acc x => if x < acc then x else acc) pi
  maxθ - minθ

-- Critical points from Lemma 1
-- Basic critical points (always exist)
def criticalPoint1 : Kuramoto3State := {θ1 := 0.0, θ2 := 0.0, θ3 := 0.0}
def criticalPoint2 : Kuramoto3State := {θ1 := 0.0, θ2 := 0.0, θ3 := pi}
def criticalPoint3 : Kuramoto3State := {θ1 := pi, θ2 := pi, θ3 := pi}
def criticalPoint4 : Kuramoto3State := {θ1 := pi, θ2 := pi, θ3 := 0.0}

-- Conditional critical points (exist when K1 = -K2)
def criticalPoint5 (params : CouplingParams) : Option Kuramoto3State :=
  if params.K1 == -params.K2 then
    some { θ1 := 0.0, θ2 := 2.0 * pi / 3.0, θ3 := pi / 3.0 }
  else none

def criticalPoint6 (params : CouplingParams) : Option Kuramoto3State :=
  if params.K1 == -params.K2 then
    some { θ1 := pi, θ2 := pi / 3.0, θ3 := 2.0 * pi / 3.0 }
  else none

-- Jacobian matrix for stability analysis
def jacobian3 (state : Kuramoto3State) (params : CouplingParams) :
    (Float × Float × Float × Float × Float × Float × Float × Float × Float) :=
  let θ1 := state.θ1
  let θ2 := state.θ2
  let θ3 := state.θ3
  let K1 := params.K1
  let K2 := params.K2

  -- Partial derivatives for Jacobian matrix
  let J11 := -K1 * Float.cos(θ2 - θ1) - K1 * Float.cos(θ3 - θ1)
  let J12 := K1 * Float.cos(θ2 - θ1)
  let J13 := K1 * Float.cos(θ3 - θ1)

  let J21 := K1 * Float.cos(θ1 - θ2)
  let J22 := -K1 * Float.cos(θ1 - θ2) - K2 * Float.cos(θ3 - θ2)
  let J23 := K2 * Float.cos(θ3 - θ2)

  let J31 := K1 * Float.cos(θ1 - θ3)
  let J32 := K2 * Float.cos(θ2 - θ3)
  let J33 := -K1 * Float.cos(θ1 - θ3) - K2 * Float.cos(θ2 - θ3)

  (J11, J12, J13, J21, J22, J23, J31, J32, J33)

-- Basin of attraction regions from Theorem 1
def basinRegion5 (state : Kuramoto3State) : Bool :=
  let θ1_θ3 := state.θ1 - state.θ3
  let θ2_θ3 := state.θ2 - state.θ3
  let θ1_θ2 := state.θ1 - state.θ2

  -- Conditions: -π < θ₁(0) - θ₃(0) < π/3, -π/3 < θ₂(0) - θ₃(0) < π, -4π/3 < θ₁(0) - θ₂(0) < 0
  (-pi < θ1_θ3 ∧ θ1_θ3 < pi/3.0) ∧
  (-pi/3.0 < θ2_θ3 ∧ θ2_θ3 < pi) ∧
  (-4.0*pi/3.0 < θ1_θ2 ∧ θ1_θ2 < 0.0)

def basinRegion6 (state : Kuramoto3State) : Bool :=
  let θ1_θ3 := state.θ1 - state.θ3
  let θ2_θ3 := state.θ2 - state.θ3
  let θ1_θ2 := state.θ1 - state.θ2

  -- Conditions: -7π/3 < θ₁(0) - θ₃(0) < -π, -π < θ₂(0) - θ₃(0) < π/3, -2π < θ₁(0) - θ₂(0) < -2π/3
  (-7.0*pi/3.0 < θ1_θ3 ∧ θ1_θ3 < -pi) ∧
  (-pi < θ2_θ3 ∧ θ2_θ3 < pi/3.0) ∧
  (-2.0*pi < θ1_θ2 ∧ θ1_θ2 < -2.0*pi/3.0)

-- Test function to verify basin regions
def testBasinRegions : IO Unit := do
  let testStates := [
    {θ1 := 0.0, θ2 := 0.0, θ3 := 0.0},  -- Should be in region 5
    {θ1 := pi, θ2 := pi/3.0, θ3 := 2.0*pi/3.0},  -- Should be in region 6
    {θ1 := pi/2.0, θ2 := pi/2.0, θ3 := pi/2.0}   -- May not be in either
  ]

  for state in testStates do
    let in5 := basinRegion5 state
    let in6 := basinRegion6 state
    IO.println s!"State: θ1={state.θ1}, θ2={state.θ2}, θ3={state.θ3}"
    IO.println s!"  In basin 5: {in5}"
    IO.println s!"  In basin 6: {in6}"
    IO.println s!"  Order parameter: {orderParameter3 state}"
    IO.println s!"  Phase diameter: {phaseDiameter state}"
    IO.println ""

-- Main function for testing
def main : IO Unit := do
  IO.println "Three-Oscillator Kuramoto System Formalization"
  IO.println "=============================================="

  let params := {K1 := -1.0, K2 := 1.0}  -- K1 = -K2 = -1.0
  IO.println s!"Coupling parameters: K1={params.K1}, K2={params.K2}"

  -- Test critical points
  IO.println "\nCritical Points:"
  IO.println s!"Θ₁*: θ1={criticalPoint1.θ1}, θ2={criticalPoint1.θ2}, θ3={criticalPoint1.θ3}"
  IO.println s!"Θ₂*: θ1={criticalPoint2.θ1}, θ2={criticalPoint2.θ2}, θ3={criticalPoint2.θ3}"
  IO.println s!"Θ₃*: θ1={criticalPoint3.θ1}, θ2={criticalPoint3.θ2}, θ3={criticalPoint3.θ3}"
  IO.println s!"Θ₄*: θ1={criticalPoint4.θ1}, θ2={criticalPoint4.θ2}, θ3={criticalPoint4.θ3}"

  let cp5 := criticalPoint5 params
  let cp6 := criticalPoint6 params
  match cp5 with
  | some pt => IO.println s!"Θ₅*: θ1={pt.θ1}, θ2={pt.θ2}, θ3={pt.θ3}"
  | none => IO.println "Θ₅*: does not exist (K1 ≠ -K2)"

  match cp6 with
  | some pt => IO.println s!"Θ₆*: θ1={pt.θ1}, θ2={pt.θ2}, θ3={pt.θ3}"
  | none => IO.println "Θ₆*: does not exist (K1 ≠ -K2)"

  -- Test basin regions
  IO.println "\nTesting Basin Regions:"
  testBasinRegions
