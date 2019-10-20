import numpy as np
from numpy.linalg import inv

def big_h(u, v):
    """The big h-function according to 139 of the specs.
    """
    left = (u + v + np.exp(-u - v)) / 2
    diff = np.abs(u - v)
    right = (diff + np.exp(-diff)) / 2
    return left - right

def big_g(alfa, q, nrofcoup, t2, tau): 
    """This function calculates 2 outputs:
     output1: g(alfa)-tau where g(alfa) is according to 165 of the specs
     output(2): gamma = Qb according to 156 of the specs
    """
    n, m = q.shape 
    # construct m*m h-matrix
    h = np.fromfunction(lambda i, j: big_h(alfa * (i+1) / nrofcoup, alfa * (j+1) / nrofcoup), (m, m))
    # Solving b according to 156 of specs, note: p = 1 by construction
    res_1 = np.array([1 - np.sum(q[i]) for i in range(n)]).reshape((n, 1))
    # b = ((Q'HQ)^(-1))(1-Q'1) according to 156 of the specs
    b = np.matmul(inv(np.matmul(np.matmul(q, h), q.T)), res_1)
    # gamma variable is used to store Qb according to 156 of specs
    gamma = np.matmul(q.T, b)
    res_2 = sum(gamma[i, 0] * (i+1) / nrofcoup for i in range(0, m))
    res_3 = sum(gamma[i, 0] * np.sinh(alfa * (i+1) / nrofcoup) for i in range(0, m))
    kappa = (1 + alfa * res_2) / res_3
    return (alfa / np.abs(1 - kappa * np.exp(t2 * alfa)) - tau, gamma)

def optimal_alfa(alfa, q, nrofcoup, t2, tau, precision):

    new_alfa, gamma = big_g(alfa, q, nrofcoup, t2, tau)
    if new_alfa > 0: 
        # scanning for the optimal alfa is based on the scan-procedure taken from Eiopa matlab production code
        # in each for-next loop the next optimal alfa decimal is scanned for, starting with an stepsize of 0.1 (first decimal) followed
        # by the a next decimal through stepsize = stepsize/10
        stepsize = 0.1
#        for alfa in range(alfamin + stepsize, 20, stepsize):
        for i in range(0, 200):
            alfa = alfa + stepsize + i / 10
            new_alfa, gamma = big_g(alfa, q, nrofcoup, t2, tau)
            if new_alfa <= 0:
                break
        for i in range(0, precision - 1):
            alfa = alfa - stepsize
            for i in range(1, 9):
                alfa = alfa + stepsize / 10
                new_alfa, gamma = big_g(alfa, q, nrofcoup, t2, tau)
                if new_alfa <= 0:
                    break
            stepsize = stepsize / 10
    return (alfa, gamma)

#Function SmithWilsonBruteForce2(Instrument As String, LiqMat As Range,
#    RatesIn As Range, nrofcoup As Integer, CRA As Double, UFRac As Double,
#    alfamin As Double, Tau As Double, T2 As Integer) As Variant
#
#    ' Input:
#    ' Instrument = {"Zero", "Bond", "Swap"}, nrofcoup = Number of Coupon
#    Payments per Year, CRA= Credit Risk Adjustment in basispoints, UFRac =
#    Ultimate Forward Rate annual commpounded (perunage), T2 = Convergence
#    Maturity
#    ' DataIn = range (50 rows x 3 columns)
#    ' Column1: Indicator Vector indicating if corresponding maturity is DLT
#    to qualify as credible input
#    ' Column2: Maturity Vector
#    ' Column3: Rate Vector
#
#    With Application.WorksheetFunction
#   
#        Dim time As Double
#        time = Timer
#        Dim data()
#        data = RangeToArray2(LiqMat, RatesIn)

#        Dim nrofrates As Integer
#        nrofrates = .Sum(.Index(data, 0, 1)) 'Sum first column of data to
#    determine number of liquid input rates
#        ReDim u(1 To nrofrates) As Double 'row vector containing liquid
#    maturities
#        ReDim r(1 To nrofrates) As Double 'row vector containing liquid rates
#        Dim umax As Double
#        
#        Dim i As Integer, j As Integer
#        j = 0
#        For i = 1 To UBound(data, 1)
#            If data(i, 1) Then                      ' Indicator = 1 <=>
#    liquid maturity/rate
#                j = j + 1
#                u(j) = data(i, 2)                   ' store liquid maturity
#    in u-vector
#                r(j) = data(i, 3) - CRA / 10000     ' store liquid rate
#    including cra in r-vector
#            End If
#        Next
#        umax = .Max(u)                              ' maximum liquid maturity
#       
#    ' Note: prices of all instruments are set to 1 by construction
#    ' Hence 1: if Instrument = Zero then for Zero i there is only one
#    pay-off of (1+r(i))^u(i) at time u(i)
#    ' Hence 2: if Instrument = Swap or Bond then for Swap/Bond i there are
#    pay-offs of r(i)/nrofcoup at time 1/nrofcoup, 2/nrofcoup, ...
#    u(i)-1/nrofcoup plus a final pay-off of 1+r(i)/nrofcoup at time u(i)
#       
#        Dim lnUFR As Double
#        lnUFR = Log(1 + UFRac)
#        'Creating Q' matrix according to 146 of specs;
#        If Instrument = "Zero" Then nrofcoup = 1
#        ReDim Q(1 To nrofrates, 1 To umax * nrofcoup) As Double
#        Select Case Instrument
#            Case "Zero" 'nrofcoup = 1 by definition
#                For i = 1 To nrofrates
#                    Q(i, u(i)) = Exp(-lnUFR * u(i)) * ((1 + r(i)) ^ u(i))
#                Next i
#            Case "Swap", "Bond"
#                For i = 1 To nrofrates
#                    For j = 1 To u(i) * nrofcoup - 1
#                        Q(i, j) = Exp(-lnUFR * j / nrofcoup) * r(i) / nrofcoup
#                    Next j
#                    Q(i, j) = Exp(-lnUFR * j / nrofcoup) * (1 + r(i) / nrofcoup)
#                Next i
#        End Select
#       
#        Dim alfa As Double
#        Dim galfa_output As Variant, gamma As Variant
#        
#        Const precision = 6 'number of decimals for optimal alfa
#        Dim alfascanoutput As Variant
#        Dim stepsize As Double
#       
##        Tau = Tau / 10000 'As input is in basispoints it has to be
 #   transformed to number format
 #      
 #       galfa_output = Galfa(alfamin, Q, nrofrates, umax, nrofcoup, T2, Tau)
 #       ' This function calculates 2 outputs:
 #       ' output(1): g(alfa)-tau where g(alfa) is according to 165 of the specs
#        ' output(2): gamma = Qb according to 148 of the specs
#
#        If galfa_output(1) <= 0 Then 'g(alfa)<=tau => alfamin is optimal
#            alfa = alfamin
#        gamm    a = galfa_output(2)
#        Else ' scanning for the optimal alfa is based on the scan-procedure
#    taken from Eiopa matlab production code
#            ' in each for-next loop the next optimal alfa decimal is
#    scanned for, starting with an stepsize of 0.1 (first decimal) followed
#    by the a next decimal through stepsize = stepsize/10
#            stepsize = 0.1
#            For alfa = alfamin + stepsize To 20 Step stepsize
#                If Galfa(alfa, Q, nrofrates, umax, nrofcoup, T2, Tau)(1) <=
#    0 Then Exit For
#            Next alfa
#            For i = 1 To precision - 1
#                alfascanoutput = AlfaScan(alfa, stepsize, Q, nrofrates,
#    umax, nrofcoup, T2, Tau)
#                alfa = alfascanoutput(1)
#                stepsize = stepsize / 10
#            Next i
#            gamma = alfascanoutput(2)
#        End If
#       
#        ' Optimal alfa and corresponding gamma have been determined
#        ' Now the SW-present value function according to 154 of the specs
#    can be calculated: p(v)=exp(-lnUFR*v)*(1+H(v,u)*Qb)
#       
#        ReDim h(0 To 121, 1 To umax * nrofcoup) As Double 'The H(v,u) matrix
#    for maturities v = 1 to 121 according to 139 of the technical specs
#    (Note: maturity 121 will not be used; see later)
#        ReDim g(0 To 121, 1 To umax * nrofcoup) As Double 'The G(v,u) matrix
#    for maturities v = 1 to 121 according to 142 of the technical specs
##    (Note: maturity 121 will not be used; see later)
 #       For i = 0 To 121
 #           For j = 1 To umax * nrofcoup
 #               h(i, j) = Hmat(alfa * i, alfa * j / nrofcoup)
 #               If (j / nrofcoup) > i Then
 #                   g(i, j) = alfa * (1 - Exp(-alfa * j / nrofcoup) *
 #   .Cosh(alfa * i))
 #               Else
 #                   g(i, j) = alfa * Exp(-alfa * i) * .Sinh(alfa * j / nrofcoup)
 #               End If
 #           Next j
 #       Next i
 #      
 #       
 #       Dim temptempdiscount
 #       Dim temptempintensity
#         ReDim tempdiscount(0 To 121) As Double
#         ReDim tempintensity(0 To 121) As Double
       
#         ReDim discount(0 To 121) As Double
#         ReDim fwintensity(0 To 121) As Double
#         ReDim yldintensity(0 To 121) As Double
#         ReDim forwardac(0 To 121) As Double
#         ReDim zeroac(0 To 121) As Double
#         ReDim forwardcc(0 To 121) As Double
#         ReDim zerocc(0 To 121) As Double

#         temptempdiscount = .Transpose(.MMult(h, gamma)) 'First a temporary
#     discount-vector will be used to store the in-between result H(v,u)*Qb =
#     H(v,u)*gamma (see 154 of the specs)
#         temptempintensity = .Transpose(.MMult(g, gamma)) 'Calculating
#     G(v,u)*Qb according to 158 of the specs
#         For i = 0 To 121
#             tempdiscount(i) = temptempdiscount(i + 1)
#             tempintensity(i) = temptempintensity(i + 1)
#         Next i
       
#         Dim temp As Double
#         temp = 0
#         For i = 1 To umax * nrofcoup
#             temp = temp + (1 - Exp(-alfa * i / nrofcoup)) * gamma(i, 1)
#     'calculating (1'-exp(-alfa*u'))Qb as subresult for 160 of specs
#         Next i
#         yldintensity(0) = lnUFR - alfa * temp 'calculating 160 of the specs
#         fwintensity(0) = yldintensity(0)
#         discount(0) = 1
#         yldintensity(1) = lnUFR - Log(1 + tempdiscount(1)) 'calculating 158
#     of the specs for maturity 1 year
#         fwintensity(1) = lnUFR - tempintensity(1) / (1 + tempdiscount(1))
#     'calculating 158 of the specs for maturity 1 year
#         discount(1) = Exp(-lnUFR) * (1 + tempdiscount(1)) 'Then the
#     in-between result from the discount vector will be used to store the
#     right discount factor into the discount vector
#         zeroac(1) = 1 / discount(1) - 1 'The default output format for
#     calculated rates is Annual Compounding
#         forwardac(1) = zeroac(1)
#         For i = 2 To 120
#                 yldintensity(i) = lnUFR - Log(1 + tempdiscount(i)) / i
#         'calculating 158 of the specs for higher maturities
#                 fwintensity(i) = lnUFR - tempintensity(i) / (1 +
#     tempdiscount(i)) 'calculating 158 of the specs for higher maturities
#             discount(i) = Exp(-lnUFR * i) * (1 + tempdiscount(i)) 'The
#     in-between result from the temporary discount vector will be used to
#     store the right discount factor into the discount vector
#             zeroac(i) = (1 / discount(i)) ^ (1 / i) - 1             'The
#     default output format for calculated rates is Annual Compounding
#             forwardac(i) = discount(i - 1) / discount(i) - 1        'The
#     default output format for calculated rates is Annual Compounding
#         Next
#         'After the last Next i=121 => now the last row of the output will be
#     (ab)used to store one ohter output item: optimal alfa
        
#         yldintensity(i) = 0
#         fwintensity(i) = 0
#         zeroac(i) = 0
#         forwardac(i) = 0
#         discount(i) = alfa 'storing optimal alfa
       
#         For i = 1 To 120
#             forwardcc(i) = Log(1 + forwardac(i))
#             zerocc(i) = Log(1 + zeroac(i))
#         Next
       
       
#         ReDim output(6)
#         output(1) = discount
#         output(2) = yldintensity
#         output(3) = zeroac
#         output(4) = fwintensity
#         outputsut(5) = forwardcc
#         output(6) = forwardac
           
#     SmithWilsonBruteForce2 = .Transpose(output)
#     End With
# End Function

def q_matrix(instrument, n, m, liquid_maturities, RatesIn, nrofcoup, cra, log_ufr):
    # Note: prices of all instruments are set to 1 by construction
    # Hence 1: if Instrument = Zero then for Zero i there is only one pay-off of (1+r(i))^u(i) at time u(i)
    # Hence 2: if Instrument = Swap or Bond then for Swap/Bond i there are pay-offs of r(i)/nrofcoup at time 1/nrofcoup, 2/nrofcoup, ...
    # u(i)-1/nrofcoup plus a final pay-off of 1+r(i)/nrofcoup at time u(i)
    q = np.zeros([n, m])
    if instrument == "Zero":
        for i, u in enumerate(liquid_maturities):
            q[i, u - 1] = np.exp(-log_ufr * u) * np.power(1 + RatesIn[u], u)
    elif instrument == "Swap" or instrument == "Bond":
        # not yet correct
        for i in range(0, n):
            for j in range(1, u[i] * nrofcoup - 1):
                q[i, j] = np.exp(-log_ufr * j / nrofcoup) * (r[i] - cra) / nrofcoup
            q[i, j] = np.exp(-log_ufr * j / nrofcoup) * (1 + (r[i] - cra) / nrofcoup)

    return q

def smith_wilson_brute_force(instrument, 
                             liquid_maturities, 
                             RatesIn, 
                             nrofcoup, 
                             cra, 
                             ufr, 
                             min_alfa, 
                             tau, 
                             T2,
                             precision = 6):

    # Input:
    # Instrument = {"Zero", "Bond", "Swap"},
    # nrofcoup = Number of Coupon Payments per Year,
    # CRA = Credit Risk Adjustment in basispoints,
    # UFRac = Ultimate Forward Rate annual commpounded (perunage),
    # T2 = Convergence Maturity
    # DataIn = range (50 rows x 3 columns)
    # Column1: Indicator Vector indicating if corresponding maturity is DLT to qualify as credible input
    # Column2: Maturity Vector

    assert (instrument == "Zero" and nrofcoup ==1),"instrument is zero bond, but with nrofcoup unequal to 1"

    n = len(liquid_maturities)               # the number of liquid rates
    m = nrofcoup * max(liquid_maturities)    # nrofcoup * maximum liquid maturity

    log_ufr = np.log(1 + ufr)
    tau = tau / 10000 
    cra = cra / 10000

    # Q' matrix according to 146 of specs;
    q = q_matrix(instrument, n, m, liquid_maturities, RatesIn, nrofcoup, cra, log_ufr)

    # Determine optimal alfa with corresponding gamma
    alfa, gamma = optimal_alfa(min_alfa, q, nrofcoup, T2, tau, precision)

    # Now the SW-present value function according to 154 of the specs can be calculated: p(v)=exp(-lnUFR*v)*(1+H(v,u)*Qb)
    #The G(v,u) matrix for maturities v = 1 to 121 according to 142 of the technical specs (Note: maturity 121 will not be used; see later)
    g = np.fromfunction(lambda i, j: np.where(j / nrofcoup > i, 
                                              alfa * (1 - np.exp(-alfa * j / nrofcoup) * np.cosh(alfa * i)), 
                                              alfa * np.exp(-alfa * i) * np.sinh(alfa * j / nrofcoup)), (121, m))

    # The H(v,u) matrix for maturities v = 1 to 121 according to 139 of the technical specs
    # h[i, j] = big_h(alfa * i, alfa * (j+1) / nrofcoup) -> strange, is different from earlier def
    h = np.fromfunction(lambda i, j: big_h(alfa * i / nrofcoup, alfa * (j+1) / nrofcoup), (122, m))

    # First a temporary discount-vector will be used to store the in-between result H(v,u)*Qb = #(v,u)*gamma (see 154 of the specs)
    temptempdiscount = np.matmul(h, gamma)
    # Calculating G(v,u)*Qb according to 158 of the specs
    temptempintensity = np.matmul(g, gamma)

    tempdiscount = np.zeros(121)
    tempintensity = np.zeros(121)
    for i in range(0, 121):
        tempdiscount[i] = temptempdiscount[i]
        tempintensity[i] = temptempintensity[i]

    #calculating (1'-exp(-alfa*u'))Qb as subresult for 160 of specs
    res1 = sum((1 - np.exp(-alfa * (i + 1) / nrofcoup)) * gamma[i, 0] for i in range(0, m))

    # yield intensities
    yldintensity = np.zeros(121)
    yldintensity[0] = log_ufr - alfa * res1 # calculating 160 of the specs
    yldintensity[1:] = log_ufr - np.log(1 + tempdiscount[1:]) / np.arange(1, 121) # calculating 158 of the specs for maturity 1 year

    # forward intensities # calculating 158 of the specs for higher maturities
    fwintensity = np.zeros(121)
    fwintensity[0] = log_ufr - alfa * res1 # calculating 160 of the specs
    fwintensity[1:] = log_ufr - tempintensity[1:] / (1 + tempdiscount[1:])  

    # discount rates 
    discount = np.zeros(121)
    discount[0] = 1
    discount[1:] = np.exp(-log_ufr * np.arange(1,121)) * (1 + tempdiscount[1:])

    # forward rates annual compounding
    forwardac = np.zeros(121)
    forwardac[0] = 0
    forwardac[1:] = discount[:-1] / discount[1:] - 1

    # zero rates annual compounding
    zeroac = np.zeros(121)
    zeroac[0] = 0
    zeroac[1:] = np.power(discount[1:], -1 / np.arange(1, 121)) - 1

    return alfa, zeroac

#Function RangeToArray2(LiqMat As Range, RatesIn As Range)
#    Dim Arr(), Arr1(), Arr2()
#    Arr1 = .Transpose(LiqMat)
#    Arr2 = .Transpose(RatesIn)
#    Dim i, length As Integer
#    length = UBound(Arr1)
#    ReDim Arr(length, 3)
#    For i = 1 To length
#        Arr(i, 1) = Arr1(i, 1)
#        Arr(i, 2) = Arr1(i, 2)
#        Arr(i, 3) = Arr2(i, 1)
#    Next i
#    RangeToArray2 = Arr
#End Function

def RangeToArray2(LiqMat, RatesIn):
    arr1 = LiqMat.T
    arr2 = RatesIn.T
    length = UBound(Arr1)
    arr = np.zeros([length, 3])
    for i in range(1, length):
        arr[i, 1] = arr1[i, 1]
        arr[i, 2] = arr1[i, 2]
        arr[i, 3] = arr2[i, 1]
    return arr

# Function FromParToForwards(ParRates2 As Range, Maturities2 As Range,
#     Span As Integer, MaxRuns As Integer, MaxError As Double) As Variant
    
#     Dim DF(), sumDF(), forwards(), ForwardsSpan(), Maturities(), ParRates()
#     Dim DiscValPlusDeriv() As Double
#     Dim i, j, k, nForwards As Integer
#     Dim f As Double

#     'ParRates = RangeToArray(ParRates2)
#     With Application.WorksheetFunction
#         ParRates = .Transpose(.Transpose(ParRates2))
#     End With
#     Maturities = RangeToArray(Maturities2)

#     nForwards = UBound(Maturities) - 1

#     ReDim DF(0 To Span)
#     ReDim sumDF(0 To Span)
#     ReDim forwards(nForwards)
#     ReDim ForwardsSpan(Span)

#     j = 1
#     DF(0) = 1
#     sumDF(0) = 0
#     For j = 1 To nForwards
#             f = 0
#             DiscValPlusDeriv =
#     DiscountedValue4par2forwards(sumDF(Maturities(j)), DF(Maturities(j)),
#     ParRates(j), f, Maturities(j + 1) - Maturities(j))
#             k = 0
#             Do While Not (Abs(DiscValPlusDeriv(1)) < MaxError) And k <= MaxRuns
#                 f = f - DiscValPlusDeriv(1) / DiscValPlusDeriv(2)
#                 DiscValPlusDeriv =
#     DiscountedValue4par2forwards(sumDF(Maturities(j)), DF(Maturities(j)),
#     ParRates(j), f, Maturities(j + 1) - Maturities(j))
#                 k = k + 1
#             Loop
#             forwards(j) = f
#         For i = Maturities(j) + 1 To Maturities(j + 1)
#             ForwardsSpan(i) = f
#             DF(i) = DF(i - 1) / (1 + ForwardsSpan(i))
#             sumDF(i) = sumDF(i - 1) + DF(i)
#         Next i
#     Next j

#     For i = Maturities(nForwards + 1) To Span
#         ForwardsSpan(i) = ForwardsSpan(i - 1)
#         DF(i) = DF(i - 1) / (1 + ForwardsSpan(i))
#         sumDF(i) = sumDF(i - 1) + DF(i)
#     Next i

#     FromParToForwards = ForwardsSpan

# End Function

def FromParToForwards(ParRates2, Maturities2, Span, MaxRuns, MaxError):
    
    # ParRates = RangeToArray(ParRates2)
    #ParRates = .Transpose(.Transpose(ParRates2))
    ParRates = ParRates2

    Maturities = RangeToArray(Maturities2)

    nForwards = UBound(Maturities) - 1

    DF = np.zero(Span)
    sumDF = np.zeros(Span)
    forwards = np.zeros(nForwards)
    ForwardsSpan = np.zeros(Span)

    j = 1
    DF[0] = 1
    sumDF[0] = 0
    for j in range(1, nForwards):
        f = 0
        DiscValPlusDeriv = DiscountedValue4par2forwards(sumDF(Maturities[j]), DF(Maturities[j]), ParRates[j], f, Maturities[j + 1] - Maturities[j])
        k = 0
        while (not (Abs(DiscValPlusDeriv(1)) < MaxError)) and (k <= MaxRuns):
            f = f - DiscValPlusDeriv[1] / DiscValPlusDeriv[2]
            DiscValPlusDeriv = DiscountedValue4par2forwards(sumDF(Maturities[j]), DF(Maturities[j]), ParRates[j], f, Maturities[j + 1] - Maturities[j])
            k = k + 1
        forwards[j] = f

    for i in range(Maturities[j] + 1, Maturities[j + 1]):
        ForwardsSpan[i] = f
        DF[i] = DF[i - 1] / (1 + ForwardsSpan[i])
        sumDF[i] = sumDF[i - 1] + DF[i]

    for i in range(Maturities(nForwards + 1), Span):
        ForwardsSpan[i] = ForwardsSpan[i - 1]
        DF[i] = DF[i - 1] / (1 + ForwardsSpan[i])
        sumDF[i] = sumDF[i - 1] + DF[i]

    return ForwardsSpan

# Function DiscountedValue4par2forwards(ByVal sumDF As Double, ByVal
#     lastDF As Double, ByVal ParRate As Double, ByVal ForwardRate As Double,
#     ByVal TminK As Integer) As Double()

#         Dim DiscVal(1 To 2) As Double
#         Dim i, k As Integer
       
#         DiscVal(1) = sumDF * ParRate
#         DiscVal(2) = 0
       
#         For i = 1 To TminK
#             DiscVal(1) = DiscVal(1) + ParRate * lastDF / ((1 + ForwardRate) ^ i)
#             DiscVal(2) = DiscVal(2) - i * ParRate * lastDF / ((1 +
#     ForwardRate) ^ (i + 1))
#         Next i
       
#         DiscVal(1) = DiscVal(1) + lastDF / ((1 + ForwardRate) ^ TminK) - 1
#         DiscVal(2) = DiscVal(2) - TminK * lastDF / ((1 + ForwardRate) ^
#     (TminK + 1))
       
#         DiscountedValue4par2forwards = DiscVal

# End Function

def DiscountedValue4par2forwards(sumDF, lastDF, ParRate, ForwardRate, TminK):

#    Dim DiscVal(1 To 2) As Double
    DiscVal = np.zeros(2)

    DiscVal[0] = sumDF * ParRate
    DiscVal[1] = 0
       
    for i in range(1, TminK):
        DiscVal[0] = DiscVal[0] + ParRate * lastDF / ((1 + ForwardRate) ^ i)
        DiscVal[1] = DiscVal[1] - i * ParRate * lastDF / ((1 + ForwardRate) ^ (i + 1))
        DiscVal[0] = DiscVal[0] + lastDF / ((1 + ForwardRate) ^ TminK) - 1
        DiscVal[1] = DiscVal[1] - TminK * lastDF / ((1 + ForwardRate) ^ (TminK + 1))

    return DiscVal

# Function BootstrapSwapToZero(SwapMatsInit As Range, SwapRatesInit As
#     Range, ZeroMaturity As Integer, Compounding As String) As Variant

#     'Bron: HenkJan vd Well  01-12-2015

#     'SwapInit bevat in de eerste kolom de oplopende looptijden van de
#     swaprates uit de tweede kolom
#     'Swaplooptijden moeten beginnen bij 1 anders valt er niks te bootstrappen

#     Dim SwapMats()
#     Dim SwapRates()

#     With Application.WorksheetFunction
#     SwapMats = .Transpose(.Transpose(SwapMatsInit))
#     SwapRates = .Transpose(.Transpose(SwapRatesInit))

#     Dim forward() As Double
#     ReDim forward(120)
#     Dim discount() As Double
#     ReDim discount(120)
#     Dim zero() As Double
#     ReDim zero(120)
#     Dim sumdiscount As Double
#     Dim m As Integer
#     Dim k As Integer
#     Dim i As Integer
#     Dim fwtemp As Double
#     Dim swaptemp As Double
     
#     m = 0
#     forward(1) = SwapRates(1)
#     zero(1) = forward(1)
#     discount(1) = 1 / (1 + forward(1))
#     sumdiscount = discount(1)

#     Dim t As Integer
#     For t = 2 To UBound(SwapMats)
#         If SwapMats(t) = SwapMats(t - 1) + 1 Then 'aansluitende swap looptijden
#             discount(SwapMats(t)) = (1 - sumdiscount * SwapRates(t)) / (1 +
#     SwapRates(t))
#             forward(SwapMats(t)) = discount(SwapMats(t - 1)) /
#     discount(SwapMats(t)) - 1
#             zero(SwapMats(t)) = (1 / discount(SwapMats(t))) ^ (1 /
#     SwapMats(t)) - 1
#             sumdiscount = sumdiscount + discount(SwapMats(t))
#             If SwapMats(t) = ZeroMaturity Then GoTo Answer
#         Else 'niet aansluitende swap looptijden => constante forward
#     interpolatie: constante forward numeriek bepalen mbv Newton Raphson
#             m = SwapMats(t) - SwapMats(t - 1)
#             fwtemp = NewtonRaphsonForward(forward(t - 1), SwapRates(t), m,
#     (1 - SwapRates(t) * sumdiscount) / discount(SwapMats(t - 1)))
#             For k = 1 To m
#                 forward(SwapMats(t - 1) + k) = fwtemp
#                 discount(SwapMats(t - 1) + k) = discount(SwapMats(t - 1) + k
#     - 1) / (1 + fwtemp)
#                 zero(SwapMats(t - 1) + k) = (1 / discount(SwapMats(t - 1) +
#     k)) ^ (1 / (SwapMats(t - 1) + k)) - 1
#                 sumdiscount = sumdiscount + discount(SwapMats(t - 1) + k)
#             Next
#             If SwapMats(t - 1) + m >= ZeroMaturity Then GoTo Answer
#         End If
       
#     Next
#     If SwapMats(t - 1) < ZeroMaturity Then 'eventueel nog extrapoleren na
#     hoogste ingegeven swaplooptijd
#         For k = 1 To ZeroMaturity - SwapMats(t - 1)
#             forward(SwapMats(t - 1) + k) = forward(SwapMats(t - 1) + k - 1)
#             discount(SwapMats(t - 1) + k) = discount(SwapMats(t - 1) + k -
#     1) / (1 + forward(SwapMats(t - 1) + k))
#             zero(SwapMats(t - 1) + k) = (1 / discount(SwapMats(t - 1) + k))
#     ^ (1 / (SwapMats(t - 1) + k)) - 1
#         Next
#     End If

#     Answer:
#     If Compounding = "C" Then
#         For i = 1 To ZeroMaturity
#             forward(i) = Log(1 + forward(i))
#             zero(i) = Log(1 + zero(i))
#         Next
#     End If

#     Dim output()
#     ReDim output(3)
#     output(1) = zero
#     output(2) = forward
#     output(3) = discount
#     output = .Transpose(output)
#     BootstrapSwapToZero = output

#     End With

# End Function

def BootstrapSwapToZero(SwapMatsInit, SwapRatesInit, ZeroMaturity, Compounding):

    #Bron: HenkJan vd Well  01-12-2015

    # SwapInit bevat in de eerste kolom de oplopende looptijden van de swaprates uit de tweede kolom
    # Swaplooptijden moeten beginnen bij 1 anders valt er niks te bootstrappen
    #SwapMats = .Transpose(.Transpose(SwapMatsInit))
    #SwapRates = .Transpose(.Transpose(SwapRatesInit))
    SwapMats = 0 # TODO
    SwapRates = 0 # TODO

    forward = np.zeros(120)
    discount = np.zeros(120)
    zero = np.zeros(120)
    m = 0
    forward[1] = SwapRates[1]
    zero[1] = forward[1]
    discount[1] = 1 / (1 + forward[1])
    sumdiscount = discount[1]

    for t in range(2, UBound(SwapMats)):
        if SwapMats(t) == SwapMats(t - 1) + 1: # aansluitende swap looptijden
            discount[SwapMats[t]] = (1 - sumdiscount * SwapRates[t]) / (1 + SwapRates[t])
            forward[SwapMats[t]] = discount[SwapMats[t - 1]] / discount[SwapMats[t]] - 1
            zero[SwapMats[t]] = (1 / discount[SwapMats[t]]) ^ (1 / SwapMats[t]) - 1
            sumdiscount = sumdiscount + discount[SwapMats[t]]
            if SwapMats[t] == ZeroMaturity:
                ww = 0
                # Then GoTo Answer
        else: 
            # niet aansluitende swap looptijden => constante forward interpolatie: 
            #constante forward numeriek bepalen mbv Newton Raphson
            m = SwapMats(t) - SwapMats(t - 1)
            fwtemp = NewtonRaphsonForward(forward[t - 1], SwapRates[t], m, (1 - SwapRates[t] * sumdiscount) / discount[SwapMats[t - 1]])
            for k in range(1, m):
                forward[SwapMats[t - 1] + k] = fwtemp
                discount[SwapMats[t - 1] + k] = discount[SwapMats[t - 1] + k - 1] / (1 + fwtemp)
                zero[SwapMats[t - 1] + k] = (1 / discount[SwapMats[t - 1] + k]) ^ (1 / (SwapMats[t - 1] + k)) - 1
                sumdiscount = sumdiscount + discount[SwapMats[t - 1] + k]
            if SwapMats[t - 1] + m >= ZeroMaturity:
                ww = 0
                # Then GoTo Answer

    if SwapMats[t - 1] < ZeroMaturity: # eventueel nog extrapoleren na hoogste ingegeven swaplooptijd
        for k in range (1, ZeroMaturity - SwapMats[t - 1]):
            forward[SwapMats[t - 1] + k] = forward[SwapMats[t - 1] + k - 1]
            discount[SwapMats[t - 1] + k] = discount[SwapMats[t - 1] + k - 1] / (1 + forward[SwapMats[t - 1] + k])
            zero[SwapMats[t - 1] + k] = (1 / discount[SwapMats[t - 1] + k]) ^ (1 / (SwapMats[t - 1] + k)) - 1

    #Answer:
    #If Compounding = "C" Then
        #For i = 1 To ZeroMaturity
            #forward(i) = Log(1 + forward(i))
            #zero(i) = Log(1 + zero(i))
        #Next
    #End If
    
    return [zero,forward,discount].T

# Function NewtonRaphsonForward(fwguess As Double, swapt1, m, c) As Double
#     Dim fw As Double
#     fw = fwguess
#     Dim fx As Double
#     Dim dfx As Double
#     Dim i As Integer
#     Dim temp As Double
   
#     For i = 1 To 500
#         temp = (1 + fw) ^ -m
#         fx = swapt1 * (1 - temp) / fw + temp - c
#         temp = temp / (1 + fw)
#         dfx = swapt1 * ((1 + (m + 1) * fw) * temp - 1) / (fw ^ 2) - m * temp
#         If Abs(fx) < 0.0000000001 Then Exit For
#         fw = fw - fx / dfx
#     Next i
#     NewtonRaphsonForward = fw
# End Function

def NewtonRaphsonForward(fwguess, swapt1, m, c):

    fw = fwguess
   
    for i in range(1, 500):
        temp = (1 + fw) ^ -m
        fx = swapt1 * (1 - temp) / fw + temp - c
        temp = temp / (1 + fw)
        dfx = swapt1 * ((1 + (m + 1) * fw) * temp - 1) / (fw ^ 2) - m * temp
        if np.abs(fx) < 0.0000000001:
            break
        fw = fw - fx / dfx
    return fw
