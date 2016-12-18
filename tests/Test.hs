module Main (
	main,
	trainUntil', learnUnary, learnBinary
	) where

import Prelude.Unicode

import Control.Concurrent
import Control.Monad
import Control.Monad.Loops
import Control.Monad.State

import Numeric.Trainee.Neural
import Numeric.LinearAlgebra

nnet ∷ IO (Net Double)
nnet = net $ input 2 ⭃ fc sigma 2 ⭃ fc sigma 2 ⭃ fc sigma 1

(⇢) ∷ a → b → Sample a b
(⇢) = (,)

samples ∷ [(Vector Double, Vector Double)]
samples = [
	vector [0, 0] ⇢ vector [0],
	vector [1, 1] ⇢ vector [0],
	vector [1, 0] ⇢ vector [1],
	vector [0, 1] ⇢ vector [1]]

main ∷ IO ()
main = do
	n ← nnet
	(e, n') ← runLearnT n $ trainUntil 1.0 10000 4 0.0001 squared samples
	liftIO $ putStrLn $ "error: " ++ show e
	liftIO $ putStrLn "result:"
	liftIO $ forM_ samples $ \(x, _) →
		putStrLn $ show x ++ " -> " ++ show (eval n' x)

trainUntil' ∷ Rational → Int → Vector Double → Cost (Vector Double) → [Sample (Vector Double) (Vector Double)] → Net Double → IO (Net Double)
trainUntil' λ batch eps c xs n = liftM snd $ runLearnT n $ iterateUntil (< eps) $ do
	xs' ← shuffleList xs
	e ← liftM avg $ replicateM 10 $ trainEpoch λ c $ makeBatches batch xs'
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
