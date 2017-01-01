{-# LANGUAGE FlexibleContexts #-}

module Main (
	main,
	trainUntil', learnUnary, learnBinary
	) where

import Prelude.Unicode

import Control.Monad
import Control.Monad.Loops
import Control.Monad.State
import Test.Hspec

import Numeric.Trainee.Neural
import Numeric.LinearAlgebra

import DataSet

main ∷ IO ()
main = hspec $
	describe "training neural network" $ do
		it "should approximate xor function" testXor
		it "should classify objects" testClassify

testXor ∷ IO ()
testXor = do
	n ← nnet
	(e, n') ← runLearnT n $ trainUntil 1.0 10000 4 1e-4 squared samples
	e `shouldSatisfy` (≤ 1e-4)
	mapM_ (shouldPass n' 0.1) samples

testClassify ∷ IO ()
testClassify = do
	n ← classNet
	let
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
		mapM_ (shouldPass n' 0.1) $ [(xs, fun xs) |
			xs ← map vector (replicateM 4 [0.0, 1.0])]

nnet ∷ IO (Net Double)
nnet = net $ input 2 ⭃ fc sigma 2 ⭃ fc sigma 2 ⭃ fc sigma 1

(⤞) ∷ (Container Vector a, Container Vector b) ⇒ [a] → [b] → Sample (Vector a) (Vector b)
xs ⤞ ys = (fromList xs, fromList ys)


samples ∷ [Sample (Vector Double) (Vector Double)]
samples = [
	[0, 0] ⤞ [0],
	[1, 1] ⤞ [0],
	[1, 0] ⤞ [1],
	[0, 1] ⤞ [1]]


classNet ∷ IO (Net Double)
classNet = net $ input 4 ⭃ fc sigma 4 ⭃ fc sigma 2 ⭃ fc sigma 1

readBalloonSamples ∷ FilePath → IO [Sample (Vector Double) (Vector Double)]
readBalloonSamples fpath = parseFile fpath (inputs ⇢ bool) where
	inputs = [
		enumValue ["yellow", "purple"],
		enumValue ["small", "large"],
		enumValue ["stretch", "dip"],
		enumValue ["adult", "child"]]

shouldPass ∷ Net Double → Double → Sample (Vector Double) (Vector Double) → IO ()
shouldPass n ε (xs, ys) = when (err > ε) $ expectationFailure msg where
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
		smps = args `zip` map fn' args
	trainUntil' 1.0 10 1e-4 squared smps n

learnBinary ∷ (Double → Double → Double) → IO (Net Double)
learnBinary fn = do
	n ← net $ input 2 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 5 ⭃ fc sigma 1
	let
		fn' v = vector [fn (v ! 0) (v ! 1)]
		args = map vector $ replicateM 2 [0.0, 0.1 .. 1.0]
		smps = args `zip` map fn' args
	trainUntil' 1.0 10 1e-4 squared smps n
