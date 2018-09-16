def minCount(val, opr):
	i = 0
	for el in lista:
		if opr == '<=':
			if el <= val:
				i+=1
		elif opr == '==':
			if el == val:
				i+=1
		else:
			if el >= val:
				i+=1
	return i