{-# LANGUAGE FlexibleContexts, OverloadedLists #-}

module Main (
	main,
	trainUntil', learnUnary, learnBinary
	) where

import Prelude.Unicode

import Control.Monad
import Control.Monad.Loops
import Control.Monad.State
import Test.Hspec
import Numeric.LinearAlgebra

import Numeric.Trainee.Data
import Numeric.Trainee.Neural

main ∷ IO ()
main = hspec $
	describe "training neural network" $ do
		it "should approximate xor function" testXor
		it "should classify objects" testClassify
		it "should support 3 classes" testIris

testXor ∷ IO ()
testXor = do
	n ← net (input 2 ⭃ fc sigma 2 ⭃ fc sigma 2 ⭃ fc sigma 1) ∷ IO (Net Double)
	(e, n') ← runLearnT n $ trainUntil 1.0 10000 4 1e-4 squared samples
	e `shouldSatisfy` (≤ 1e-4)
	mapM_ (shouldPass n' 0.1) samples

testClassify ∷ IO ()
testClassify = do
	n ← net (input 4 ⭃ fc sigma 4 ⭃ fc sigma 2 ⭃ fc sigma 1) ∷ IO (Net Double)
	let
		cases ∷ [(String, Vector Double → Vector Double)]
		cases = [
			("adult+stretch", fn $ \[_, _, act, age] → act ≡ 0.0 ∨ age ≡ 0.0),
			("adult-stretch", fn $ \[_, _, act, age] → act ≡ 0.0 ∧ age ≡ 0.0),
			("yellow-small+adult-stretch", fn $ \[color, sz, act, age] → (color ≡ 0.0 ∧ sz ≡ 0.0) ∨ (act ≡ 0.0 ∧ age ≡ 0.0)),
			("yellow-small", fn $ \[color, sz, _, _] → color ≡ 0.0 ∧ sz ≡ 0.0)]
		fn ∷ ([Double] → Bool) → Vector Double → Vector Double
		fn f = vector ∘ return ∘ fromIntegral ∘ fromEnum ∘ f ∘ toList
	forM_ cases $ \(name, fun) → do
		classes ← readBalloonSamples $ "data/classify/balloon/" ++ name ++ ".data"
		(e, n') ← runLearnT n $ trainUntil 1.0 1000 10 1e-4 squared classes
		e `shouldSatisfy` (≤ 1e-4)
		mapM_ (shouldPass n' 0.1) [xs ⇢ fun xs |
			xs ← map vector (replicateM 4 [0.0, 1.0])]

testIris ∷ IO ()
testIris = do
	n ← net (input 4 ⭃ fc sigma 4 ⭃ fc sigma 4 ⭃ fc sigma 3) ∷ IO (Net Double)
	classes ← parseCsvFile "data/classify/iris/iris.data" (inputs ⇢ outs)
	(e, n') ← runLearnT n $ trainUntil 5.0 10000 10 1e-5 squared classes
	e `shouldSatisfy` (≤ 1e-4)
	mapM_ (shouldPass n' 0.1) classes
	where
		inputs ∷ [Attr String Double]
		inputs = [
			read_ `onAttr` scale 0.1,
			read_ `onAttr` scale 0.1,
			read_ `onAttr` scale 0.1,
			read_ `onAttr` scale 0.1]
		outs ∷ [Attr String Double]
		outs = [class_ ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]]


samples ∷ [Sample (Vector Double) (Vector Double)]
samples = [
	[0, 0] ⇢ [0],
	[1, 1] ⇢ [0],
	[1, 0] ⇢ [1],
	[0, 1] ⇢ [1]]

readBalloonSamples ∷ FilePath → IO [Sample (Vector Double) (Vector Double)]
readBalloonSamples fpath = parseCsvFile fpath (inputs ⇢ [bool]) where
	inputs = [
		enum_ ["yellow", "purple"],
		enum_ ["small", "large"],
		enum_ ["stretch", "dip"],
		enum_ ["adult", "child"]]
	bool = enum_ ["f", "t"]

shouldPass ∷ Net Double → Double → Sample (Vector Double) (Vector Double) → IO ()
shouldPass n ε (Sample xs ys) = when (err > ε) $ expectationFailure msg where
	msg = show xs ++ " -> " ++ show res ++ " should be " ++ show ys
	res = eval n xs
	err = vecSize (res - ys)
	vecSize v = sqrt (dot v v)

-- not used

trainUntil' ∷ Rational → Int → Vector Double → Cost (Vector Double) → [Sample (Vector Double) (Vector Double)] → Net Double → IO (Net Double)
trainUntil' λ batch eps c xs n = fmap snd $ runLearnT n $ iterateUntil (< eps) $ do
	xs' ← shuffleList xs
	e ← fmap avg $ replicateM 10 $ trainEpoch λ c $ makeBatches batch xs'
	liftIO $ print e
	return e

learnUnary ∷ (Double → Double) → IO (Net Double)
learnUnary fn = do
	n ← net $ input 1 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 1
	let
		fn' v = vector [fn (v ! 0)]
		args = map vector $ replicateM 1 [0.0, 0.1 .. 1.0]
		smps = zipWith (⇢) args (map fn' args)
	trainUntil' 1.0 10 1e-4 squared smps n

learnBinary ∷ (Double → Double → Double) → IO (Net Double)
learnBinary fn = do
	n ← net $ input 2 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 1
	let
		fn' v = vector [fn (v ! 0) (v ! 1)]
		args = map vector $ replicateM 2 [0.0, 0.1 .. 1.0]
		smps = zipWith (⇢) args (map fn' args)
	trainUntil' 1.0 10 1e-4 squared smps n
