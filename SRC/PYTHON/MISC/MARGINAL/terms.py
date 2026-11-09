import sympy as sp
import argparse

from fractions import Fraction
from collections import defaultdict

delta = sp.symbols("delta", real=True)
I = sp.I

psi1p, psi1m, psi2p, psi2m = sp.symbols(
	"psi1p psi1m psi2p psi2m",
	commutative=False,
)

psi1p_dagger, psi1m_dagger, psi2p_dagger, psi2m_dagger = sp.symbols(
	"psi1p_dagger psi1m_dagger psi2p_dagger psi2m_dagger",
	commutative=False,
)

R1, L1, R2, L2 = sp.symbols(
	"R1 L1 R2 L2",
	commutative=False,
)

R1_dagger, L1_dagger, R2_dagger, L2_dagger = sp.symbols(
	"R1_dagger L1_dagger R2_dagger L2_dagger",
	commutative=False,
)

K = sp.symbols(
	"K",
	real=True,
)

c = sp.cos(K / 2)
s = sp.sin(K / 2)

modes = [
	((Fraction(1, 4), +1), psi1p, psi1p_dagger),
	((Fraction(-1, 4), -1), psi1m, psi1m_dagger),
	((Fraction(3, 4), -1), psi2p, psi2p_dagger),
	((Fraction(-3, 4), +1), psi2m, psi2m_dagger),
]

patch_canonical_order = [
	psi1p_dagger,
	psi1m_dagger,
	psi2p_dagger,
	psi2m_dagger,
	psi1p,
	psi1m,
	psi2p,
	psi2m,
]

patch_order_index = {
	op: index
	for index, op in enumerate(patch_canonical_order)
}

canonical_order = [
	R1_dagger,
	L1_dagger,
	R2_dagger,
	L2_dagger,
	R1,
	L1,
	R2,
	L2,
]

order_index = {
	op: index
	for index, op in enumerate(canonical_order)
}

bogoliubov = {
	psi1p: [
		(c, L1),
		(-I * s, R1_dagger),
	],
	psi1p_dagger: [
		(c, L1_dagger),
		(+I * s, R1),
	],
	psi1m: [
		(+I * s, L1_dagger),
		(c, R1),
	],
	psi1m_dagger: [
		(-I * s, L1),
		(c, R1_dagger),
	],
	psi2p: [
		(s, R2),
		(-I * c, L2_dagger),
	],
	psi2p_dagger: [
		(s, R2_dagger),
		(+I * c, L2),
	],
	psi2m: [
		(+I * c, R2_dagger),
		(s, L2),
	],
	psi2m_dagger: [
		(-I * c, R2),
		(s, L2_dagger),
	],
}


def zero_momentum():

	return (Fraction(0), 0)


def add_momentum(momentum_1, momentum_2):

	return (
		momentum_1[0] + momentum_2[0],
		momentum_1[1] + momentum_2[1],
	)


def multiply_momentum(momentum, integer):

	return (
		integer * momentum[0],
		integer * momentum[1],
	)


def minus_momentum(momentum):
	return multiply_momentum(momentum, -1)


def momentum_expression(momentum):

	pi_coefficient, delta_coefficient = momentum

	pi_coefficient = sp.Rational(
		pi_coefficient.numerator,
		pi_coefficient.denominator,
	)

	return pi_coefficient * sp.pi + delta_coefficient * delta


def is_nonoscillating(momentum):

	pi_coefficient, delta_coefficient = momentum

	if delta_coefficient != 0:
		return False

	if pi_coefficient.denominator != 1:
		return False

	return pi_coefficient.numerator % 2 == 0


def fermion_terms(offset):

	annihilation_terms = []
	creation_terms = []

	for momentum, operator, operator_dagger in modes:
		annihilation_terms.append((
			momentum,
			multiply_momentum(momentum, offset),
			operator,
			sp.Integer(1),
		))

		opposite_momentum = minus_momentum(momentum)

		creation_terms.append((
			opposite_momentum,
			multiply_momentum(opposite_momentum, offset),
			operator_dagger,
			sp.Integer(1),
		))

	return annihilation_terms, creation_terms


def xi_terms(offset):

	annihilation_terms, creation_terms = fermion_terms(offset)

	return creation_terms + annihilation_terms


def eta_terms(offset):

	annihilation_terms, creation_terms = fermion_terms(offset)
	terms = []

	for phase_j, phase_offset, operator, coefficient in creation_terms:
		terms.append((
			phase_j,
			phase_offset,
			operator,
			+ I * coefficient,
		))

	for phase_j, phase_offset, operator, coefficient in annihilation_terms:
		terms.append((
			phase_j,
			phase_offset,
			operator,
			- I * coefficient,
		))

	return terms


def multiply_expansions(expansions):

	terms = [(
		zero_momentum(),
		zero_momentum(),
		tuple(),
		sp.Integer(1),
	)]

	for expansion in expansions:
		new_terms = []

		for phase_j_1, phase_offset_1, operator_1, coefficient_1 in terms:
			for phase_j_2, phase_offset_2, operator_2, coefficient_2 in expansion:
				new_terms.append((
					add_momentum(phase_j_1, phase_j_2),
					add_momentum(phase_offset_1, phase_offset_2),
					operator_1 + (operator_2,),
					coefficient_1 * coefficient_2,
				))

		terms = new_terms

	return terms


def canonicalize(operators, current_order_index):

	operators = list(operators)
	sign = 1

	for left in range(len(operators)):
		for right in range(left + 1, len(operators)):
			if current_order_index[operators[left]] > current_order_index[operators[right]]:
				sign *= -1

	operators.sort(key=lambda operator: current_order_index[operator])

	return sign, tuple(operators)


def simplify_coefficient(coefficient):

	coefficient = sp.expand_complex(coefficient)
	coefficient = sp.expand_trig(coefficient)
	coefficient = sp.trigsimp(
		coefficient,
		method="fu",
	)

	coefficient = coefficient.rewrite(sp.sin)
	coefficient = sp.trigsimp(
		coefficient,
		method="fu",
	)

	coefficient = coefficient.subs(
		K,
		sp.pi / 4 + delta,
	)

	coefficient = sp.expand_trig(coefficient)
	coefficient = sp.cancel(coefficient)
	coefficient = sp.trigsimp(
		coefficient,
		method="fu",
	)

	return sp.factor_terms(
		sp.simplify(coefficient)
	)


def collect_terms(terms, prefactor):

	result = defaultdict(lambda: sp.Integer(0))

	for phase_j, phase_offset, operators, coefficient in terms:
		if not is_nonoscillating(phase_j):
			continue

		if len(set(operators)) != len(operators):
			continue

		sign, ordered_operators = canonicalize(
			operators,
			patch_order_index,
		)

		coefficient = (
			prefactor
			* sign
			* coefficient
			* sp.exp(I * momentum_expression(phase_offset))
		)

		result[ordered_operators] += coefficient

	clean_result = {}

	for operators, coefficient in result.items():
		coefficient = simplify_coefficient(coefficient)

		if coefficient != 0:
			clean_result[operators] = coefficient

	return clean_result


def add_results(result_1, result_2):

	result = defaultdict(lambda: sp.Integer(0))

	for operators, coefficient in result_1.items():
		result[operators] += coefficient

	for operators, coefficient in result_2.items():
		result[operators] += coefficient

	clean_result = {}

	for operators, coefficient in result.items():
		coefficient = simplify_coefficient(coefficient)

		if coefficient != 0:
			clean_result[operators] = coefficient

	return clean_result


def apply_bogoliubov(result):

	transformed_result = defaultdict(lambda: sp.Integer(0))

	for operators, coefficient in result.items():
		terms = [(
			tuple(),
			coefficient,
		)]

		for operator in operators:
			new_terms = []

			for transformed_operators, transformed_coefficient in terms:
				for bogoliubov_coefficient, bogoliubov_operator in bogoliubov[operator]:
					new_terms.append((
						transformed_operators + (bogoliubov_operator,),
						transformed_coefficient * bogoliubov_coefficient,
					))

			terms = new_terms

		for transformed_operators, transformed_coefficient in terms:
			if len(set(transformed_operators)) != len(transformed_operators):
				continue

			sign, ordered_operators = canonicalize(
				transformed_operators,
				order_index,
			)

			transformed_result[ordered_operators] += (
				sign * transformed_coefficient
			)

	clean_result = {}

	for operators, coefficient in transformed_result.items():
		coefficient = simplify_coefficient(coefficient)

		if coefficient != 0:
			clean_result[operators] = coefficient

	return clean_result


dagger = {
	R1: R1_dagger,
	L1: L1_dagger,
	R2: R2_dagger,
	L2: L2_dagger,
	R1_dagger: R1,
	L1_dagger: L1,
	R2_dagger: R2,
	L2_dagger: L2,
}


def sorted_terms(result):

	return sorted(
		result.items(),
		key=lambda item: tuple(
			order_index[operator]
			for operator in item[0]
		),
	)


def operator_string(operators):

	return " ".join(str(operator) for operator in operators)


def print_result(result):

	coefficient_width = max(
		len(sp.sstr(coefficient))
		for coefficient in result.values()
	)

	for index, (operators, coefficient) in enumerate(
		sorted_terms(result),
		start=1,
	):
		print(
			f"[{index:02d}] "
			f"{sp.sstr(coefficient):<{coefficient_width}} "
			f" |  {operator_string(operators)}"
		)


def main():

	term_1 = multiply_expansions([
		xi_terms(0),
		eta_terms(0),
		xi_terms(1),
		eta_terms(1),
	])

	term_2 = multiply_expansions([
		eta_terms(0),
		eta_terms(1),
		xi_terms(2),
		xi_terms(3),
	])

	result_1 = collect_terms(term_1, prefactor=-1)
	result_2 = collect_terms(term_2, prefactor=+1)

	patch_result = add_results(result_1, result_2)
	result = apply_bogoliubov(patch_result)

	print_result(result)

if __name__ == "__main__":
	main()
