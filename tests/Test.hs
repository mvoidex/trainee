{-# LANGUAGE FlexibleContexts, OverloadedLists, OverloadedStrings, UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Main (
	main,
	testXor, testClassify, testIris, testMnist
	) where

import Prelude.Unicode

import Control.DeepSeq
import Control.Monad
import Control.Monad.State.Strict
import Control.Monad.Morph
import qualified Data.Attoparsec.Text as A
import qualified Data.Text.IO as T
import qualified Data.Vector as V
import Test.Hspec
import Numeric.LinearAlgebra hiding (conv2, conv)
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
	(e, n') ← runLearnT n $ trainUntil 1.0 10000 4 1e-4 squared xorSamples
	e `shouldSatisfy` (≤ 1e-4)
	mapM_ (shouldPass n' 0.1) xorSamples

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
	(_, n') ← runLearnT n $ hoist (`evalStateT` (rightAnswers n classes, 0)) $ learnIris classes
	let
		failedSamples = V.filter (not ∘ rightClass n' ∘ snd) $ V.zip (V.fromList ([1..] ∷ [Integer])) classes
	putStrLn "--- Done. ---"
	putStrLn "Failed samples:"
	forM_ failedSamples $ \(i, Sample inp outp) → putStrLn $ "{n}\t{input} → {output} ≢ {right}"
		~~ ("n" ~% i)
		~~ ("input" ~% inp)
		~~ ("output" ~% eval n' inp)
		~~ ("right" ~% outp)
	length failedSamples `shouldSatisfy` (≤ 15)
	where
		learnIris ∷ Samples (Vector Double) (Vector Double) → StateT (Net Double) (StateT (Int, Int) IO) ()
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


testMnist ∷ IO ()
testMnist = do
	n ← net (input 784 ⭃ conv2 sigma 28 (3, 3) ⭃ conv2 sigma 26 (3, 3) ⭃ fc sigma 32 ⭃ fc sigma 10)
	putStrLn "reading train data"
	smps ← readMnist "data/classify/mnist/train.csv"
	putStrLn $ "loaded {0} samples" ~~ length smps
	(_, n') ← runLearnT n $ learnMnist smps
	print n'
	where
		learnMnist ∷ Samples (Vector Double) (Vector Double) → StateT (Net Double) IO ()
		learnMnist smps = do
			let
				ixs = takeWhile (< V.length smps) $ map (* 20) [0 .. ]
				bs = map (\ix → V.slice ix (min 20 (V.length smps - ix)) smps) ixs
			liftIO $ putStrLn $ "prepared {0} batches" ~~ length bs
			es ← forM bs $ \b → do
				e ← trainBatch 0.01 crossEntropy b
				liftIO $ putStrLn $ "batch error: {e}" ~~ ("e" ~% e)
				return e
			let
				e = avg es
			liftIO $ putStrLn $ "error: {e}" ~~ ("e" ~% e)
			when (e > 0.1) $ learnMnist smps


rightClass ∷ Net Double → Sample (Vector Double) (Vector Double) → Bool
rightClass n_ (Sample i o) = maxIndex (eval n_ i) ≡ maxIndex o

rightAnswers ∷ Net Double → Samples (Vector Double) (Vector Double) → Int
rightAnswers n_ = V.length ∘ V.filter (rightClass n_)


xorSamples ∷ Samples (Vector Double) (Vector Double)
xorSamples = samples [
	[0, 0] ⇢ [0],
	[1, 1] ⇢ [0],
	[1, 0] ⇢ [1],
	[0, 1] ⇢ [1]]

readIrisData ∷ FilePath → IO (Samples (Vector Double) (Vector Double))
readIrisData fpath = parseCsvFile False fpath $ do
	is ← mapM (col_ >=> read_ >=> (return ∘ (* 0.1))) [0 .. 3]
	os ← col_ 4 >>= class_ ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
	sample_ is os

readBalloonSamples ∷ FilePath → IO (Samples (Vector Double) (Vector Double))
readBalloonSamples fpath = parseCsvFile False fpath $ do
	is ← sequence [
		col_ 0 >>= enum_ ["yellow", "purple"],
		col_ 1 >>= enum_ ["small", "large"],
		col_ 2 >>= enum_ ["stretch", "dip"],
		col_ 3 >>= enum_ ["adult", "child"]]
	os ← col_ 4 >>= enum_ ["f", "t"] >>= single_
	sample_ is os

readMnist ∷ FilePath → IO (Samples (Vector Double) (Vector Double))
readMnist fpath = do
	cts ← T.readFile fpath
	either error return $ A.parseOnly mnist cts
	where
		mnist ∷ A.Parser (Samples (Vector Double) (Vector Double))
		mnist = do
			_ ← A.manyTill A.anyChar A.endOfLine
			fmap samples $ A.many' $ do
				(f:fs) ← A.sepBy A.decimal (A.char ',') <* A.endOfLine
				let
					is = fromList $ map ((/ 255.0) ∘ fromIntegral) fs
					os = fromList $ replicate f 0.0 ++ [1.0] ++ replicate (10 - f - 1) 0.0
				is `deepseq` os `deepseq` return (Sample is os)

shouldPass ∷ Net Double → Double → Sample (Vector Double) (Vector Double) → IO ()
shouldPass n ε (Sample xs ys) = when (err > ε) $ expectationFailure msg where
	msg = show xs ++ " -> " ++ show res ++ " should be " ++ show ys
	res = eval n xs
	err = vecSize (res - ys)
	vecSize v = sqrt (dot v v)
