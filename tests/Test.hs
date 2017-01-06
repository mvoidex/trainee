{-# LANGUAGE FlexibleContexts, OverloadedLists, OverloadedStrings, UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Main (
	main,
	testXor, testClassify, testIris
	) where

import Prelude.Unicode

import Control.Monad
import Control.Monad.State.Strict
import Control.Monad.Morph
import Test.Hspec
import Numeric.LinearAlgebra
import Text.Format

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
		classes ← readBalloonSamples $ "data/classify/balloon/{name}.data" ~~ ("name" ~% name)
		(e, n') ← runLearnT n $ trainUntil 1.0 1000 10 1e-4 squared classes
		e `shouldSatisfy` (≤ 1e-4)
		mapM_ (shouldPass n' 0.1) [xs ⇢ fun xs |
			xs ← map vector (replicateM 4 [0.0, 1.0])]


instance Show (Vector a) ⇒ FormatBuild (Vector a)


testIris ∷ IO ()
testIris = do
	n ← net (input 4 ⭃ fc sigma 12 ⭃ fc sigma 3) ∷ IO (Net Double)
	classes ← readIrisData "data/classify/iris/iris.data"
	(_, n') ← runLearnT n $ hoist (flip evalStateT (rightAnswers n classes, 0)) $ learnIris classes
	let
		failedSamples = filter ((> 0.2) ∘ miss n' crossEntropy ∘ snd) $ zip ([1..] ∷ [Integer]) classes
	putStrLn "--- Done. ---"
	putStrLn "Failed samples:"
	forM_ failedSamples $ \(i, Sample inp outp) → putStrLn $ "{n}\t{input} → {output} ≢ {right}"
		~~ ("n" ~% i)
		~~ ("input" ~% inp)
		~~ ("output" ~% eval n' inp)
		~~ ("right" ~% outp)
	length failedSamples `shouldSatisfy` (≤ 15)
	where
		learnIris ∷ [Sample (Vector Double) (Vector Double)] → StateT (Net Double) (StateT (Int, Int) IO) ()
		learnIris classes = do
			e ← fmap last $ replicateM 100 $ trainEpoch 0.01 150 crossEntropy classes
			n' ← get
			let
				ans = rightAnswers n' classes
			lift $ modify (\(ans', long') → (ans, if ans ≡ ans' then succ long' else 0))
			liftIO $ putStrLn $ "correct answers: {rights}/{total}; error = {e}"
				~~ ("e" ~% e)
				~~ ("rights" ~% ans)
				~~ ("total" ~% length classes)
			long' ← lift $ gets snd
			when (long' ≤ 10 ∧ e > 0.1) $ learnIris classes
		rightAnswers net_ samples_ = length $ filter (≤ 0.2) $ map (miss net_ crossEntropy) samples_


samples ∷ [Sample (Vector Double) (Vector Double)]
samples = [
	[0, 0] ⇢ [0],
	[1, 1] ⇢ [0],
	[1, 0] ⇢ [1],
	[0, 1] ⇢ [1]]

readIrisData ∷ FilePath → IO [Sample (Vector Double) (Vector Double)]
readIrisData fpath = parseCsvFile fpath (inputs ⇢ outs) where
	inputs ∷ [Attr String Double]
	inputs = [
		read_ `onAttr` scale 0.1,
		read_ `onAttr` scale 0.1,
		read_ `onAttr` scale 0.1,
		read_ `onAttr` scale 0.1]
	outs ∷ [Attr String Double]
	outs = [class_ ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]]

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
